"""Monsters followed from frame to frame, for the continuous feed (localization.worldstate): each one keeps
an id while it walks and turns, flinches out of sight under damage numbers, or leaves the screen and comes
back, and what happens to it comes out as events.

A frame's sightings are the whole screen's detections of that very frame (since 2026-10-30, one pass a frame at the
perception's one rate, localization.rate.HZ: 15 a second; until then each track was re-found around where it should
be in every frame at 30 a second, and a search of the whole screen every other frame was merged frames after its own).
Each is matched to a track (`update`, `_match`): same kind, same platform, and where its track should be by then
(`predicted`: walking on its heading at its kind's speed, turning back at its platform's ends) within 10 + 1.3 x its
speed x the time since + 25 units (knockback): 43.7 units a frame on at 15, against a frame's prediction 28 off at p99
on the user's own play. Greedy, nearest first (a walking track met by a detection facing the other way, away from its
platform's ends, FACING_COST farther: two of a kind passing within the gate, 2.1% of that clip's frames, keep their ids).
One no track explains is new: `entered` at the screen's edge (the camera moved, or it walked in), `spawned` near a spawn
point of its kind (or found fading in at one: perception.spawns, its alpha in the event), else `appeared`. Which
detection each track was this frame is left in `matched`.

A monster found fading in at its spawn point (the frame step's spawn watch, perception.spawns: Mob.fading its alpha,
until the whole screen finds it, a median 7 frames on) is a track like any other from its first sighting, its alpha in
it (`fading`) and when it turns solid (`solid_at`: alpha FADE_PER_S a second), carried on that line once the whole
screen has it (it finds them at alpha 0.6-0.8): a newborn can be hit while it fades, and didn't touch her until it was
solid (perception.spawns). Such a sighting is the track of a monster first seen since its fade began only, never an
older one's, nor the newborn's before it at its point (FADE_SLACK_S; until the review of 2026-11-01 a track seen fading
itself was let take it: 4 times on the clips the next newborn at a point 7 s later took the last one's hidden track);
and its track isn't dropped as a flicker (MIN_SEEN_S): the watch's sightings are a monster's (none false on the labelled
clips), lost now and then for a frame or two under cover before the whole screen finds them (68 of 390).

A track's place is its last detection's feet; its velocity is no measure of its own (the user, 2026-10-30: "Perception
should only give mob heading, speed should come from data-mined game files"): its kind's speed from the game data the
way it faces in a move frame, 0 standing or flinching (monsters.walk_vx), from its first detection on.

A track not seen in a frame:
- where it should be is off screen: `offscreen` (event `left`), remembered OFFSCREEN_S to know it again
  when it comes back (`back`);
- on screen but unseen for MISSING_S: `hidden`. Hit, a monster flinches under its damage numbers and the
  detector loses it for 1-3 s; one back within HIDE_S is the same one;
- hidden for longer than HIDE_S (counted from her last attack its way, when her throws are told: every hit
  renews its damage numbers): `dead`, killed by someone else (`died`, cause "other", `by` the other player by it,
  localization.players.near_rival): no EXP gain of hers explains it. Unless it went out of sight right beside
  another of its kind, or a new one was seen where it was meanwhile: in a crowd of one kind, tracks lose their
  monsters to each other, so such a track is let go without a death (`lost in a crowd`); or (OTHERS_BY_PLAYER) no
  other player on the minimap is by it, nobody who could have killed it (`nobody by it`).

Her EXP gains (worldstate, from the EXP number each frame) are kills of hers: each kill a gain's size was, of the
kind it names when it names one (localization.gains; a gain no kill explains isn't one). The game never says which
monster died; her attack took it as its key went down, and the gain shows GAIN_AFTER_KEY (0.1-0.6 s) after that key,
while the monster is still drawn alive, walking with its bar up, until her star lands (live on 2026-10-26, by eye:
its first damage numbers 0.5 s after the gain, its death frames from 0.6 s, gone at 1.2 s). So a gain that soon after
a throw's key is the kill of the first of its kind that attack takes from where she was then, by where the tracks are
at the gain (`_picks`: the nearest ahead of her from VICTIM_UP above her feet to VICTIM_DOWN below, those seen lately
first, out-of-sight ones where they should be): that track is bound to the gain at once (`gain`), `dying`, followed
as it walks on until her star lands on it (STAR_LANDS_S: no farther than its kind walks till then, nor after it), its
id in every state since (Monster.track), so whoever reads the state knows which monster is dead, where it walks, until
it goes; `dead`, cause "hers", once out of sight CONFIRM_S after the gain, or at once when its death is seen
(perception.deaths, `seen_dying`). One seen again still REBIND_S after the gain was another: let go, and the gain goes on
the next its attack took that's gone, or one seen dying DEATH_AFTER_GAIN_S after the gain or later. A gain no throw
explains so (none keyed in the window before it; `watch`, her throws unknown) waits as
before, FADE_S + MISSING_S, for the monster it was to go out of sight: one she threw at (a throw toward it in the
STAR_WINDOW before; with her throws unknown any monster within a star's reach of her), of the kind named, out of sight
by then, the likeliest first: one that stood still where it was from the gain until it went, then one that went after
the gain (else one already out of sight when it came: killed while it hid), then (PLAYERS_LAST) one with no other
player by it, then the least HP left, then the nearest her: `dead`, cause "hers"; meanwhile the likeliest one still
seen is `dying`, and the likeliest of all, if out of sight already, `owed` (localization.unseen doesn't take it for a
monster about). Where her attack picks by the client's own search and no star flies (her Shape's `kill`: the
magician's Magic Claw, from 2026-11-02), a gain is the kill of the first in the search's bands from where she was as
the cast picked, the monster her attack choice aimed it at first in that band (`throw`'s aim), out-of-search ones
after them all (`_band_picks`); its monster walks on only until her hit shows, her KillCredit's lands_s after the key;
let go, it goes first on one in her search whose death was seen since (BAND_LET_GO_DEAD). A share of a kill (a gain
no kill explains: another player's kill of a monster her stars hurt) can go on the monster hurt that her throws went
at (`share`, SHARES: off), dying and dead as a kill of hers is, its cause "other". A monster's HP bar going down is
a `bar_hit`, by her when a throw of hers went its way ("other" otherwise; None with her throws unknown).

A track seen in fewer frames than MIN_SEEN_S makes at the perception's rate that goes is dropped without a death: a
match on scenery for a frame or two isn't a monster someone killed. Nothing grows without bound: dead tracks go
DEAD_KEEP_S after their death, offscreen ones after OFFSCREEN_S, gains and throws after their windows.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from maplehelper.engine import RIGHT

from ..character import THIEF_SHAPE, Shape
from ..perception.kinds import kind_of
from ..perception.mobs import Mob
from ..perception.spawns import FADE_PER_S
from .mapdata import MapModel
from .monsters import (DEFAULT_SPEED, GAIN_AFTER_KEY, MONSTER_HALF, MONSTER_HEIGHT, Monster, bodies, speeds,
                       walk_edges, walk_vx)
from .players import Player, near_rival
from .rate import HZ

LIVE, HIDDEN, OFFSCREEN, DYING, DEAD = "live", "hidden", "offscreen", "dying", "dead"
STATES = (LIVE, HIDDEN, OFFSCREEN, DYING, DEAD)

MISSING_S = 0.3     # s unseen on screen before a track is `hidden`: the detector misses a monster in a look now and
                    # then (the simulator's game preset: 1 look in 10), 4.5 frames at 15 a second. On the user's own
                    # play's clip the whole screen's chains of 5+ frames that went missing came back after 1 frame 83
                    # times, 2 frames 35, 3 29, 4 25, of 483 ends (D:/MapleHelper-scratch/replan/p15/clip_gaps.py)
HIDE_S = 3.5        # s a monster may stay out of sight on screen and still be the same one: hit, it hides under its
                    # damage numbers for 1-3 s (the README); longer, someone killed it
FADE_S = 1.5        # s a killed monster may still be seen fading where it died (about a second, the README; the
                    # simulator's FADE 0.8 s): a gain waits this long (and MISSING_S) for the one it was to go
EXTRAPOLATE_S = 0.5  # s an unseen monster is taken to walk on (a hit one flinches, knocked back, not walking)
# A track must have been seen in this long's frames (at the perception's rate: 3 at 15 a second) for its going to count
# as a death: 5 frames at 30 a second until 2026-10-30 (0.17 s).
MIN_SEEN_S = 0.2
OFFSCREEN_S = 10.0  # s a track off screen is remembered, to know it again when it comes back
DEAD_KEEP_S = 2.0   # s a dead track is kept (shown as dead) before it goes
MATCH_BASE, MATCH_SPEED, MATCH_KNOCKBACK = 10.0, 1.3, 25.0  # units: a sighting within 10 + 1.3 speed dt + 25
EDGE_PX = 24        # half-size px (1920x1080; View.edge at the look's geometry) from the play area's side: a new
                    # monster this close came in from off screen
SPAWN_NEAR = 60.0   # units: a new monster this close to a spawn point of its kind (on its platform), away from the
                    # screen's edges, spawned there
FADE_SLACK_S = 0.1  # s: a sighting fading in at alpha a is a track first seen no sooner than a / FADE_PER_S before it,
                    # this much give (its alpha off the line by up to 0.08 a frame: perception.spawns.DROP_A)
STAR_WINDOW = 2.5   # s after a throw's key that a kill or a hit can be its star's (behavior.ledger.STAR_WINDOW)
HIT_AFTER_S = 1.2   # s after a throw's key that its star may hit at the latest (it leaves 0.52 s after the key and
                    # flies up to 0.7 s): a monster hidden under her damage numbers stays hidden HIDE_S from then. In
                    # the simulator's game-like world a slime clawed every 0.85 s hid for 5 s and was taken for dead
# Units ahead of her, by its middle as the tracker has it at the throw's key, that a monster her star picks can
# be: 434 (attack.reach.STAR_REACH: a star picks a body whose near side is within STAR_RANGE 400, so the widest
# kind's middle up to 434), and 40 for what she and it move between the frame and the pick (physics.JUMP_LAG after
# the key) and the tracker's error. Of 724 picks by the simulator's stars in the tracker table's seven runs
# (README: Continuous perception), the farthest was 449 ahead at the throw, 3 past 434. It was 460 + 80, from
# when a star was taken to reach 460 to a body's edge: a kill could be put on a monster 500-540 off.
THROW_REACH = 434.0 + 40.0
THROW_DY = 70.0     # ... and above or below her feet (60 measured, and a little slack)
# Her kill's monster (`_picks`, `_bind`): the attack of the throw keyed GAIN_AFTER_KEY before her gain took it, as it
# picked PICK_S after the key (attack.reach: about 65 ms) from where she was then (her place in the frames, mid-jump
# for a jump throw), the nearest ahead of her of the gain's kind, VICTIM_BEHIND behind her middle at most (the claw's
# boxes start 12-20 ahead and a body reaches back half its width), VICTIM_UP above her feet then to VICTIM_DOWN below
# (the star's search windows open upward, to 124 above at the far bands: attack.reach), where each track is at the
# gain's frame, one seen in the last FRESH_S first (out of sight longer, it's where it should be, and often isn't:
# hit, hidden under her damage numbers, or gone). Against 85 of her kills on the Lie Detector's and the rolling
# recording's frames of 2026-10-25/26, labelled by eye (D:/MapleHelper-scratch/kplace/truth; the service's frame step
# replayed on them, her throws and the live gains' kinds told: kplace/build/variants.py), the truth's track was on 76
# so (89%), 9 another; the nearest within 75 up or down, 63 and 15 (7 none); within 110 up and down 68 and 17; the
# fresh first from 0.25 s 74, 0.35 s 73, 0.5 s 72; 100 up as 110, 120 up 70, 150 up 69; by the client's search
# (attack.reach.ground_pick's order) 72. Master's ledger had 52 right of the 85, the tracker's dying mark 62 (truth).
PICK_S = 0.065
VICTIM_BEHIND = 20.0
VICTIM_UP = 110.0
VICTIM_DOWN = 75.0
FRESH_S = 0.2
BIND_S = 0.3        # s after a gain its monster may still be bound (a late search's track is as of its own frame)
# A gain whose kind isn't known (its size not read, or no kill's: 154 of the 1,225 gains of 2026-10-26 18:02-22:54)
# goes on the nearest ahead of any kind, those whose HP bar shows (hurt: her throws went at them) first.
UNKNOWN_HURT_FIRST = True
# Her kill's monster where her attack picks by the client's own search (her Shape's `kill`: the magician's Magic Claw,
# from 2026-11-02; `_band_picks`): see BAND_* below and the README (Her kill's monster by Magic Claw's search).
AIM_DX = 60.0         # units: a throw's aim told by place (no track: a replay's) is the track of its kind there nearest
                      # where she saw it, this near (a monster walks 20-25 units between her choice and the key)
# Her kill let go (`_let_go`) goes first on one in her search whose death was seen since, booked another's then (the
# critique of 2026-11-02: R3 00:49:59.9's victim, its death seen and booked 'other' as it lay outside VICTIM_DOWN). On
# R1-R3 with the band order it never fired (that victim is among her picks now, and `_deaths_seen` moves the gain onto
# it as it dies); kept for the victim whose track is new since the bind (a re-found body: P16's id churn). Tried with it
# on the same replays and dropped, no first binding nor final changed on R1+R2: a gain let go with nothing out of sight
# waiting for its monster again; a death seen taking a gain only from one in her search.
BAND_LET_GO_DEAD = True
# Her star lands on the one she killed STAR_LANDS_S after its key and her distance to it over STAR_FLIGHT units a
# second more (live on 2026-10-26, the Lie Detector's recording of 23:10: 23 of 26 kills within 0.15 s of that;
# sim.world's DIE_LANDS_S, DIE_FLIGHT): until then its monster walks on, after it it stands where it got to and plays
# its death, and the frames' detector, which has no die1 pictures, loses it (a median 0.5 s after the gain). So a bound
# track is re-found (or a search's sighting taken for it) only within a track's reach with its walk stopped
# LANDS_SLACK_S after then, and once lost from sight MISSING_S by then, no farther than FADE_MOVE + MATCH_KNOCKBACK from
# its last sighting (`_stray`): one farther off is another monster her window found, a track of its own at once
# (`update`'s strays: a new one, or another's), not it nor a sign that it lives. Its reach grew as any track's with the
# time it was out of sight, and on the four recordings' replays 8 of 165 bound tracks were re-found 45-97 units from
# their last sighting 0.7-1.0 s on, 7 of them let go so, 2 of those right bindings by the labels (review of 2026-10-27,
# D:/MapleHelper-scratch/kplace/review-truth: binds.py; at 23:14:41, w2310-037, the green she killed lost 0.25 s after
# the gain and a green 56 units on re-found at 1.18 s, where the walk stopped at her star's landing took 111 units; at
# 00:58:28, w0056-015, the horny re-found 71 units on a frame after the last). Held to FADE_MOVE from the landing whether
# lost or not, the track a filter had walked on past where a green stopped lost it (w0020-099: a new track for her
# dying kill, a target again), and wrong bindings to live ones that walked on weren't let go (kplace/fix2: score.py,
# heldout.py, excl.py on the replays, tags k2-k5).
STAR_LANDS_S = 0.75
STAR_FLIGHT = 1490.0
LANDS_AFTER_GAIN_S = 0.6  # ... for a track bound with no throw known: 0.60 s after the gain at the median (diag-signs)
LANDS_SLACK_S = 0.2       # ... give or take (23 of the 26 within 0.15 s)
BOUND_LOST = True         # (False: a bound track re-found as any track)
# A death seen sooner than this after a gain isn't that gain's kill (`_deaths_seen`): her star hasn't landed. Of the 106
# deaths of hers labelled on the recordings (D:/MapleHelper-scratch/kplace/diag-signs: labels.json), die1 began 0.60 s
# after the gain at the median, 0.35 at p10, under 0.17 s (when a death seen at 0.3 would have begun, the watch seeing
# it 0.13 s in) on 2. Without it, another player's kill seen dying in the gain's frame took the gain, her own victim
# a target again (review of 2026-10-27: review-behaviour/order.py); on the recordings the two swaps came 0.53 and 1.00 s
# after their gains.
DEATH_AFTER_GAIN_S = 0.3
# A throw told again at its key the other way (behavior.hunter: her look's frame, shown after its key with no arrow or
# hit since, shows her facing that way: her keys had her facing the way the stance threw, and the game hadn't turned
# her) is that throw, its way put right (`throw`), and a gain bound off it already whose monster's death isn't seen
# goes on what that way took (`_retold`: let go, bound again). Her throws' ways are the stance's keys, and of 4,493
# attack records of 8 runs 49 had a frame after them showing her the other way; on the recording of 2026-10-26 00:20 two
# of 15 kills were bound so to a live one on the side she didn't throw at (w0020-097, -102: review of 2026-10-27,
# review-truth).
RETOLD_S = 1e-6     # s: a throw told at its key's time within this is the one told before
# A track bound to a gain is drawn alive until her star lands (0.6-0.9 s after the gain), then dying, last drawn about
# 1.2 s after the gain (by eye, 1.02-1.53 at p10-p90), and the frames' detector, which has no die1 pictures, loses it a
# median 0.5 s after the gain (p90 0.81: D:/MapleHelper-scratch/kplace/truth). So one re-found REBIND_S after its gain
# was another: let go, and the gain goes on the next her attack took that's gone (on the labels' 85 kills, of the 5
# rebinds 2 put a wrong kill right, 1 a right one wrong); one out of sight MISSING_S from CONFIRM_S on is dead.
REBIND_S = 1.2
CONFIRM_S = 1.5
# Whether a share of a kill (another player's, of a monster her stars hurt: a gain no kill explains) is bound to the
# hurt one her throws went at with the lowest bar (`_bind_share`). Emulated on the telemetry of the grinds of 2026-10-26
# 21:04-22:28 (D:/MapleHelper-scratch/kplace/build/idsim.py --ids), it took 4 more throws at the dead off their target
# but 42 more at monsters alive by the service's deaths, 14 of them seen alive again 1.5 s on; the death watch
# (perception.deaths) sees most of those deaths anyway (in the recordings 4 of the 6 throws at another player's kill
# left out without it): off.
SHARES = False
BAR_STEP = 0.03     # a bar this much lower than its lowest seen before: a hit
# A track's velocity is its detection's heading at its kind's speed from the game data (monsters.walk_vx: a move frame
# walks the way it faces, a stand or hit1 frame stands), as it was until 2026-10-30 for every sighting whose pose was
# known (POSE_VELOCITY: monsters start, stop and turn at once, and an alpha-beta filter on x lagged 0.5 s behind one
# setting off), the filter gone with it (a sighting whose pose isn't known keeps the velocity it had). POSE_GATE_S
# (None: the pose alone, the default) takes a move frame's speed only while it left a FADE_MOVE box in the last
# POSE_GATE_S (or was first seen that recently), a stand frame's stop only while it didn't: in game the pose is the frame
# of the template that matched, and on the user's own play at 15 a second a move frame stood still over the next 0.2 s
# 11% of the time, a stand frame moved 5% (D:/MapleHelper-scratch/replan/p15, clip_motion.py), where the earlier
# recordings had 18% and 22% wrong for up to 1.4 s on end; there the pose alone predicted a frame, 0.2, 0.5 and 1 s on
# best of the rules tried (the median 0.6, 1.0, 4.0 and 21.4 units off; with a move frame taken only where it moved in
# the last 0.2 s, 0.6, 1.0, 6.0 and 23.5). Kept as a switch for the A/B. (At 0.25 s, the default until 2026-10-30, it
# cut the deaths judged someone else's and the tracks lost in a crowd in the simulator with 20% of the poses wrong in
# 1 s spells: 12, 19 against the pose alone's 34, 36.)
POSE_GATE_S: float | None = None
# A walking track met by a detection facing the other way, away from its platform's ends (FACING_END from the end its
# centre turns at), is that much farther from it in the matching: a monster turns mid-platform 0.07 times a second
# against 0.45 at an end (the user's own play, clip_turns_out.txt), and its facing is the way it walks 99.2% of the time,
# so two of a kind passing within a track's gate (2.1% of the clip's frames) keep their ids.
FACING_COST = 8.0
FACING_END = 40.0
FRAME_SLACK = 0.05  # s: a monster last seen this little before a gain went out of sight as it died
FADE_MOVE = 10.0    # units a monster fading where it died may seem to move: a sighting farther off is another one
                    # walking through (it took the dying one's track, and starved its own, in the simulator)
# Other players on the minimap (localization.players: told each frame, `update`'s `players`). With OTHERS_BY_PLAYER, a
# monster gone from sight on screen past HIDE_S is someone else's kill only with another player by it (near_rival:
# 400 units across, 100 up or down); with nobody by it, nobody else can have killed it, and it's let go without a
# death. Over the seven runs of the README's table (Continuous perception), with the other player hunting one place
# at a time: deaths judged someone else's 14 -> 4, right 4 -> 4 (10 let go, nobody by them); hers the same (263 judged,
# 254 right, of 274). With PLAYERS_LAST, of the monsters her EXP gain may be, one with another player by it comes after
# those without (as behavior.ledger.PLAYER_KILLS has hers): the same 263 there, and with her gains naming their kinds
# the same 283 judged hers (all right) of 299 on or off; on as PLAYER_KILLS is, nothing measured against it. Either way
# a death someone else's names the player by it (`by`). On the three tracks together (the feed's searches off its frame
# loop too; the README's table): 7 judged someone else's of the other player's 44, all right and named, 11 let go;
# 290 judged hers, all right, of 304.
OTHERS_BY_PLAYER = True
PLAYERS_LAST = True


@dataclass
class Event:
    """Something that happened on screen, at the frame it showed."""
    kind: str               # entered, spawned, appeared, back, left, died, bar_hit; exp_gain, hp_drop (worldstate)
    t: float                # the frame's time (when the screen showed it)
    track: int | None = None
    data: dict = field(default_factory=dict)

    def text(self) -> str:
        bits = ", ".join(f"{k} {v}" for k, v in self.data.items() if v is not None)
        return self.kind + (f" #{self.track}" if self.track is not None else "") + (f": {bits}" if bits else "")


@dataclass
class Track:
    """One monster followed from frame to frame."""
    id: int
    kind: str
    platform: int | None
    x: float                 # map units, its feet (its last detection's)
    y: float
    mob: Mob                 # its last sighting: its picture's box on screen, the template that matched
    first_seen: float
    last_seen: float
    vx: float = 0.0          # units/s: its heading at its kind's speed from the game data (monsters.walk_vx)
    facing: int = 0          # 1 right, -1 left, 0 not known (the template that matched)
    pose: str = ""
    seen: int = 1            # frames it was seen in
    bar: float | None = None  # its HP bar (fraction left) when last read; None: never seen hurt
    bar_at: float | None = None
    hits: int = 0            # times its bar went down
    state: str = LIVE
    died_at: float | None = None
    cause: str | None = None  # "hers" or "other", once dead
    judged: float | None = None  # when it was found dead
    how: str = ""            # how it was first seen (entered, spawned, appeared)
    crowded: bool = False    # it went out of sight beside another of its kind, or a new one was seen where it was
    still_x: float | None = None  # where it has stood since `still_since` (within FADE_MOVE)
    still_since: float = 0.0
    owed: bool = False       # out of sight, the one a gain of hers still waiting would go on now (_resolve): likely
                             # dead, not about (localization.unseen)
    gain: float | None = None  # the gain of hers (its frame's time) this monster is the kill of (`_bind`), or the
                               # share (`share`): dying from then, dead once gone
    share: bool = False      # ... a share: another player's kill of a monster she hurt
    lands: float | None = None  # ... and when her star lands on it (STAR_LANDS_S: it walks on till then, `_stray`)
    seen_x: float | None = None  # where its last sighting put it (map x, unfiltered: perception.deaths looks there)
    died_seen: float | None = None  # when its death was seen (perception.deaths: its die1 frames where it was)
    fading: float | None = None   # its alpha as of its last sighting, fading in at its spawn point: the spawn watch's
                                  # (Mob.fading), or on its fade's line from the last of those (FADE_PER_S); None: solid
                                  # (or never seen fading)
    solid_at: float | None = None  # when it turns solid by its fade (the watch's last sighting of it: alpha a at t,
                                   # solid at t + (1 - a) / FADE_PER_S); None: never seen fading in

    def record(self) -> dict:
        """As telemetry data ('tracks'): its `vx` the game data's speed the way it heads (since 2026-10-30, no measure:
        its walk over time is its x from record to record), its `pose` the last detection's."""
        return {"id": self.id, "kind": self.kind, "state": self.state, "x": round(self.x, 1), "y": round(self.y, 1),
                "vx": round(self.vx, 1), "facing": self.facing, "pose": self.pose, "platform": self.platform,
                "bar": None if self.bar is None else round(self.bar, 2), "hits": self.hits, "seen": self.seen,
                **({"cause": self.cause} if self.cause else {}),
                **({"gain": round(self.gain, 3)} if self.gain is not None else {}),
                **({"share": True} if self.share else {}),
                **({"fading": round(self.fading, 3)} if self.fading is not None else {})}

    def opacity(self, t: float) -> float | None:
        """Its alpha at t by its fade (FADE_PER_S from the spawn watch's last sighting of it), 1.0 once solid; None:
        never seen fading in."""
        if self.solid_at is None:
            return None
        return max(0.0, min(1.0, 1.0 - (self.solid_at - t) * FADE_PER_S))

    @property
    def killed(self) -> bool:
        """Whether it's dead or dying as far as the tracker knows: found dead, or the monster a gain of hers (or a
        share) is on."""
        return self.state == DEAD or (self.state == DYING and self.gain is not None)


@dataclass(frozen=True)
class View:
    """What a frame shows of the map (map units: x from left to right, y from top to bottom), the play area
    above the bottom bar, and its size on screen (look-frame px), to tell a monster out of sight on screen
    from one off it. A box of the map the minimap covers (`covered`) doesn't show either."""
    left: float
    right: float
    top: float
    bottom: float
    width: int                # look-frame px
    height: int
    covered: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1 (map units)
    edge: float = EDGE_PX     # look-frame px: EDGE_PX in the look's geometry

    def shows(self, kind: str, x: float, y: float) -> bool:
        """Whether a monster of that kind with its feet at (x, y) would show wholly on screen."""
        half, tall = MONSTER_HALF.get(kind, 34.0), MONSTER_HEIGHT.get(kind, 46.0)
        if not (self.left + half <= x <= self.right - half and self.top + tall <= y <= self.bottom):
            return False
        c = self.covered
        return c is None or not (c[0] < x + half and x - half < c[2] and c[1] < y and y - tall < c[3])

    def at_edge(self, mob: Mob) -> bool:
        """Whether a monster's picture is within EDGE_PX (`edge`) of the play area's sides."""
        e = self.edge
        return (mob.x - mob.w // 2 < e or mob.x + mob.w // 2 > self.width - e
                or mob.y - mob.h // 2 < e or mob.y + mob.h // 2 > self.height - e)


@dataclass
class _Throw:
    t: float
    side: str
    aimed: set[int]  # the tracks a star thrown then could meet
    her: tuple[float, float] | None = None  # where she was then (in the last frame placed)
    attack: str | None = None  # which of her attacks it was (gamelab.character.AttackSpec's id; None: not told)
    aim: int | None = None     # the track her attack choice aimed it at (`throw`'s aim, found: `_aim_of`), if told


class MonsterTracker:
    """Tracks of the monsters on screen, kept up frame by frame (`update`), and what happened to them.
    `throws`: whether her throws are told (`throw`): then a kill or a hit is hers only after a throw her way;
    without (`watch`, played by hand) an EXP gain is hers anyway, put on a monster in a star's reach. `shape`: what her
    attacks are to it (gamelab.character.Shape: which monsters a use went at, and its windows); THIEF_SHAPE, the thief's,
    is the constants above (THROW_REACH, THROW_DY, HIT_AFTER_S, STAR_WINDOW, HIDE_S)."""

    def __init__(self, model: MapModel | None = None, throws: bool = True, shape: Shape = THIEF_SHAPE):
        self.model = model
        self.shape = shape
        self.walking = speeds(model)
        # (each kind's body, what her search picks by: the map's game data, else monsters' tables)
        self.bodies = {} if shape.kill is None else bodies(model if hasattr(model, "mob_bodies") else None)
        self.spawns = [] if model is None else [
            (kind_of(s.name), model.platform_under(s.x, s.y, reach=60), s.x) for s in model.spawns if s.name]
        self.throws_known = throws
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.gains: list[tuple[float, str | None]] = []  # her kills her EXP told of, not yet put on a monster: when,
                                                         # and the kind the gain's size named (None: not known)
        self.throws: list[_Throw] = []
        self.shares: list[float] = []  # shares of kills her EXP told of (`share`), not yet put on a monster
        # Each bound gain's (or share's) monsters her attack took in turn (`_picks`, ids), for when the one it went on
        # was another (`_judge`): kept STAR_WINDOW + HIDE_S
        self.picked: dict[float, list[int]] = {}
        # ... its throw's key, her x then (when its star lands) and the gain's kind (bound again, `_retold`)
        self.keyed: dict[float, tuple[float, float, str | None]] = {}
        self.retold: list[float] = []  # the keys of throws told again the other way since the last frame (`throw`)
        self.her: tuple[float, float] | None = None  # where she was in the last frame placed
        self._last_t = -1e18                          # ... and when that was
        # where she was, frame by frame (5 s at the perception's rate)
        self._path: deque[tuple[float, float, float]] = deque(maxlen=round(5 * HZ))
        self.matched: dict[int, Monster] = {}  # the last frame's detections, by the track each was (`update`)
        self.players: list[Player] | None = None     # the other players on the map in the last frame (None: not told)
        self.counts = {"tracks": 0, "died hers": 0, "died other": 0, "lost in a crowd": 0, "flickers": 0,
                       "gains unexplained": 0, "late sightings of the dead": 0, "nobody by it": 0}
        # (and, once they happen, "kills bound", "shares bound", "let go", "deaths seen": `_count`)

    # --- told from outside -----------------------------------------------------------------------

    def gain(self, t: float, kind: str | None = None) -> None:
        """Her EXP went up in the frame shown at t, by a kill of `kind` (by the gain's size: localization.gains; None:
        not known). A gain of several kills is told once for each."""
        self.gains.append((t, kind))

    def share(self, t: float) -> None:
        """Her EXP went up in the frame shown at t by a share of a kill (a gain no kill of the map's explains:
        localization.gains, Gain.odd): another player's kill of a monster her stars hurt."""
        self.shares.append(t)

    def seen_dying(self, tid: int, t: float) -> None:
        """Track `tid`'s death was seen in the frame shown at t (perception.deaths: its die1 frames where it was)."""
        tr = self.tracks.get(tid)
        if tr is not None and tr.state != DEAD and tr.died_seen is None:
            tr.died_seen = t

    def throw(self, t: float, side: str, attack: str | None = None, aim: tuple | None = None) -> None:
        """She threw toward `side` at t (Actions.throw_listeners), with `attack` (its id, if told) and `aim`, the
        monster her attack choice aimed it at, if told ((track, kind, x, platform): its track's id, None where not
        known, and where it was seen when she chose it; `_aim_of`): the monsters a star could meet from where she was in
        the last frame. One told again at its key (RETOLD_S) another way is that throw the way she faced
        (behavior.hunter's look after it): its way and path put right, and its kill bound again at the next frame
        (`_retold`)."""
        for th in self.throws:
            if abs(th.t - t) <= RETOLD_S:
                if th.side != side:
                    th.side = side
                    th.aimed = self._aimed_by(th)
                    self.retold.append(th.t)
                return
        th = _Throw(t, side, set(), self.her, attack, self._aim_of(aim, t))
        th.aimed = self._aimed_by(th)
        self.throws.append(th)

    def _aim_of(self, aim: tuple | None, t: float) -> int | None:
        """The track a throw keyed at t was aimed at (`throw`'s `aim`): its own, where it's one of the tracks alive of
        that kind; else the one of that kind on that platform nearest where she saw it (by where each should be at t),
        within AIM_DX (a replay's aim, from the hunt's records, has no track of this tracker's: sim.replay.feed); None:
        none told or found."""
        if aim is None:
            return None
        track, kind, x, platform = (tuple(aim) + (None,) * 4)[:4]
        tr = self.tracks.get(track) if track is not None else None
        if tr is not None and tr.state != DEAD and (kind is None or tr.kind == kind):
            return tr.id
        if x is None:
            return None
        near = [(abs(self.predicted(tr, t) - x), tr.id) for tr in self.tracks.values()
                if tr.state != DEAD and (kind is None or tr.kind == kind)
                and (platform is None or tr.platform is None or tr.platform == platform)]
        best = min(near, default=None)
        return None if best is None or best[0] > AIM_DX else best[1]

    def _aimed_by(self, th: "_Throw") -> set[int]:
        """The tracks a star of throw `th` could meet (`_in_path`)."""
        return {tr.id for tr in self.tracks.values() if tr.state not in (DEAD, OFFSCREEN) and self._in_path(th, tr)}

    # --- each frame ---------------------------------------------------------------------------------

    def update(self, t: float, found: list[Monster], view: View | None, her: tuple[float, float] | None,
               players: list[Player] | None = None,
               read: Callable[[Monster, "Track | None"], Monster] | None = None,
               dying: Callable[[dict[int, Monster]], list[int]] | None = None) -> list[Event]:
        """The frame shown at t: `found`, the whole screen's detections in it, placed on the map; `view`, what it shows
        of the map (None: she isn't placed, and nothing can be judged); `her`, where she is; `players`, the other
        players on the map (localization.players; None: not known). Each detection is matched to a track (`_match`) or
        is a new one; `read(m, track)` (the world step's: its HP bar read if that's due, worldstate._bar_due; the track
        None for a new one) gives the detection as the track takes it. Which detection each track was is left in
        `matched`. `dying(matched)`: the tracks whose death this frame shows (perception.deaths, the world step's), told
        before her gains and deaths are judged. Returns what happened."""
        events: list[Event] = []
        if players is not None:
            self.players = players
        self.matched = {}
        if view is None:
            return events
        self.her = her
        if her is not None:
            self._path.append((t, her[0], her[1]))
        seen: set[int] = set()
        for m, tr in self._match(t, found):
            if read is not None:
                m = read(m, tr)
            if tr is None:
                tr = self._new(t, m, view)
                fading = {} if m.mob.fading is None else {"fading": m.mob.fading}
                events.append(Event(tr.how, t, tr.id, {"kind": m.kind, "x": round(m.x), "platform": m.platform,
                                                       **fading}))
            else:
                if tr.state == OFFSCREEN:
                    events.append(Event("back", t, tr.id, {"kind": m.kind, "after": round(t - tr.last_seen, 2)}))
                self._seen(tr, m, t, events)
            seen.add(tr.id)
            self.matched[tr.id] = m
        for tr in list(self.tracks.values()):
            if tr.id not in seen:
                self._unseen(tr, t, view, events)
        if dying is not None:
            for tid in dying(self.matched):
                self.seen_dying(tid, t)
        self._resolve(t, events)
        self.throws = [th for th in self.throws if t - th.t <= self.shape.window_s + self.shape.hide_s]
        self._last_t = t
        return events

    def live(self) -> list[Track]:
        """The tracks of monsters alive (seen or not)."""
        return [tr for tr in self.tracks.values() if tr.state != DEAD]

    def predicted(self, tr: Track, t: float) -> float:
        """Where track `tr` should be at t (map x): walking on its heading (its velocity: the game data's speed the way
        it faces, 0 standing) for EXTRAPOLATE_S at most, turning back at its platform's ends as MonsterTrack.at plans it
        (it never walks off them: 80% of the turns on the user's own play came within 40 units of the end it can reach,
        and the tracker clamped it there until 2026-10-30)."""
        return self._walked(tr, tr.vx * min(max(0.0, t - tr.last_seen), EXTRAPOLATE_S))

    # --- inside ---------------------------------------------------------------------------------------

    def _ends(self, tr: Track) -> tuple[float, float] | None:
        """The ends its centre turns at on its platform (map x: monsters.walk_edges, until 2026-10-31 its half width
        inside them), or None (no platform, or no map)."""
        if self.model is None or tr.platform is None:
            return None
        return walk_edges(self.model.platform(tr.platform))

    def _walked(self, tr: Track, go: float) -> float:
        """Track `tr`'s place `go` units walked on from its last sighting (+ right), turning back at its platform's ends
        (`_ends`)."""
        ends = self._ends(tr)
        if ends is None:
            return tr.x + go
        lo, hi = ends
        if hi <= lo:
            return lo
        k = (min(max(tr.x, lo), hi) + go - lo) % (2 * (hi - lo))
        return lo + (k if k <= hi - lo else 2 * (hi - lo) - k)

    def _at(self, tr: Track, t: float) -> float:
        """Where track `tr` was (or should be) at t, also before it was last seen: walked back, as long as
        `predicted` walks it on."""
        if t >= tr.last_seen:
            return self.predicted(tr, t)
        return self._walked(tr, -tr.vx * min(tr.last_seen - t, EXTRAPOLATE_S))

    def _match(self, t: float, found: list[Monster]) -> list[tuple[Monster, Track | None]]:
        """Each detection and the track it is (None: a new monster): nearest pairs first, one each (see the top). A
        track dying with no gain of hers on it (the likeliest of a waiting gain's, still seen fading) takes a detection
        only within FADE_MOVE of where it was: one farther is another walking through."""
        pairs = []
        for i, m in enumerate(found):
            for tr in self.tracks.values():
                if tr.kind != m.kind or tr.state == DEAD:
                    continue
                if tr.platform is not None and m.platform is not None and tr.platform != m.platform:
                    continue
                if tr.state == DYING and tr.gain is None and abs(m.x - tr.x) > FADE_MOVE:
                    continue
                if t > tr.last_seen and self._stray(tr, m, t):
                    continue  # (her kill's monster, lost where her star landed on it)
                if m.mob.fading is not None and tr.first_seen < t - m.mob.fading / FADE_PER_S - FADE_SLACK_S:
                    continue  # (a newborn isn't a monster seen before its fade began)
                x = self.predicted(tr, t)
                gap = abs(m.x - x)
                speed = self.walking.get(tr.kind, DEFAULT_SPEED)
                if gap <= MATCH_BASE + MATCH_SPEED * speed * max(0.0, t - tr.last_seen) + MATCH_KNOCKBACK:
                    pairs.append((gap + (FACING_COST if self._turned(tr, m, x) else 0.0), i, tr.id))
        out: dict[int, Track] = {}
        used: set[int] = set()
        for _, i, tid in sorted(pairs):
            if i not in out and tid not in used:
                out[i] = self.tracks[tid]
                used.add(tid)
        return [(m, out.get(i)) for i, m in enumerate(found)]

    def _turned(self, tr: Track, m: Monster, x: float) -> bool:
        """Whether detection `m` faces against walking track `tr`'s heading where `tr` should be now (x), away from the
        ends it turns at (FACING_END): a turn mid-platform, rare (FACING_COST)."""
        if not tr.vx or not m.facing or (m.facing > 0) == (tr.vx > 0):
            return False
        ends = self._ends(tr)
        return ends is None or (x - ends[0] > FACING_END and ends[1] - x > FACING_END)

    def _in_path(self, th: _Throw, tr: Track) -> bool:
        """Whether a star of throw `th` could meet track `tr`: where it should be at the throw's key, ahead of
        where she was then by THROW_REACH at most (10 behind), THROW_DY above or below her feet (her Shape's near,
        ahead, up and down)."""
        if th.her is None:
            return False
        sign = 1 if th.side == RIGHT else -1
        s = self.shape
        return (s.near <= sign * (self.predicted(tr, th.t) - th.her[0]) <= s.ahead
                and -s.up <= tr.y - th.her[1] <= s.down)

    def _stray(self, tr: Track, m: Monster, t: float) -> bool:
        """Whether sighting `m` at t, taken for track `tr`, is another monster: `tr` bound to a gain, and `m` farther
        than a track's reach with its kind's walk stopped LANDS_SLACK_S after her star lands on it (STAR_LANDS_S); lost
        from sight MISSING_S by then, farther than FADE_MOVE + MATCH_KNOCKBACK from its last sighting."""
        if not BOUND_LOST or tr.state != DYING or tr.gain is None:
            return False
        stop = (tr.lands if tr.lands is not None else tr.gain + LANDS_AFTER_GAIN_S) + LANDS_SLACK_S
        if t >= stop and t - tr.last_seen >= MISSING_S:
            return abs(m.x - (tr.seen_x if tr.seen_x is not None else tr.x)) > FADE_MOVE + MATCH_KNOCKBACK
        walk = self.walking.get(tr.kind, DEFAULT_SPEED) * max(0.0, min(t, stop) - tr.last_seen)
        return abs(m.x - tr.x) > MATCH_BASE + MATCH_KNOCKBACK + MATCH_SPEED * walk

    def _new(self, t: float, m: Monster, view: View) -> Track:
        how = "spawned" if m.mob.fading is not None else "entered" if view.at_edge(m.mob) else \
            "spawned" if self._near_spawn(m) else "appeared"
        for o in self.tracks.values():  # a hidden one where this one is was likely lost to a mix-up
            if o.state == HIDDEN and o.kind == m.kind and abs(o.x - m.x) <= 2 * (MATCH_BASE + MATCH_KNOCKBACK):
                o.crowded = True
        vx = walk_vx(m.kind, m.facing, m.pose, self.walking)
        tr = Track(self.next_id, m.kind, m.platform, m.x, m.y, m.mob, t, t, vx=0.0 if vx is None else vx,
                   facing=m.facing, pose=m.pose, bar=m.bar, bar_at=t if m.bar is not None else None, how=how,
                   seen_x=m.x)
        self._faded(tr, m, t)
        self.next_id += 1
        self.counts["tracks"] += 1
        self.tracks[tr.id] = tr
        return tr

    def _near_spawn(self, m: Monster) -> bool:
        return any(kind == m.kind and p is not None and p.id == m.platform and abs(x - m.x) <= SPAWN_NEAR
                   for kind, p, x in self.spawns)

    def _seen(self, tr: Track, m: Monster, t: float, events: list[Event]) -> None:
        """Track `tr` seen as `m` in the frame at t: where it is (its detection's feet), how fast it walks (its heading
        at its kind's speed: monsters.walk_vx; POSE_GATE_S), its bar."""
        tr.x = m.x
        # Whether its sightings say it walks: it left a FADE_MOVE box in the last POSE_GATE_S, or was first seen
        # that recently (nothing says yet that it stands). None: the pose alone decides.
        walks = None if POSE_GATE_S is None else (
            tr.still_x is None or abs(m.x - tr.still_x) > FADE_MOVE or t - tr.still_since < POSE_GATE_S)
        vx = walk_vx(m.kind, m.facing, m.pose, self.walking)
        if vx is not None and (walks is None or (walks if vx else not walks)):
            tr.vx = vx
        tr.y, tr.mob = m.y, m.mob
        if m.platform is not None:
            tr.platform = m.platform
        if m.facing:
            tr.facing = m.facing
        tr.pose = m.pose
        tr.last_seen, tr.seen, tr.crowded, tr.seen_x = t, tr.seen + 1, False, m.x
        if tr.still_x is None or abs(m.x - tr.still_x) > FADE_MOVE:
            tr.still_x, tr.still_since = m.x, t
        if tr.state in (HIDDEN, OFFSCREEN):
            tr.state = LIVE
        self._faded(tr, m, t)
        if m.bar is not None:
            if m.bar <= (tr.bar if tr.bar is not None else 1.0) - BAR_STEP:
                tr.hits += 1
                events.append(Event("bar_hit", t, tr.id, {"bar": round(m.bar, 2), "by": self._by(tr, t)}))
            tr.bar, tr.bar_at = m.bar, t

    @staticmethod
    def _faded(tr: Track, m: Monster, t: float) -> None:
        """Track `tr`'s fade as sighting `m` at t says: the spawn watch's alpha and when it turns solid by it, or (the
        whole screen's) the alpha on its fade's line until then."""
        if m.mob.fading is not None:
            tr.fading, tr.solid_at = m.mob.fading, t + (1.0 - m.mob.fading) / FADE_PER_S
        else:
            a = tr.opacity(t)
            tr.fading = None if a is None or a >= 1.0 else round(a, 3)

    def _by(self, tr: Track, t: float) -> str | None:
        """Who hit it, as far as her throws tell."""
        if not self.throws_known:
            return None
        return "hers" if any(0 <= t - th.t <= self.shape.window_s and tr.id in th.aimed for th in self.throws) \
            else "other"

    def _unseen(self, tr: Track, t: float, view: View, events: list[Event]) -> None:
        dt = t - tr.last_seen
        if tr.state == DEAD:
            if t - tr.judged > DEAD_KEEP_S:
                del self.tracks[tr.id]
            return
        if not view.shows(tr.kind, self.predicted(tr, t), tr.y):
            if tr.state not in (OFFSCREEN, DYING):
                tr.state = OFFSCREEN
                events.append(Event("left", t, tr.id, {"kind": tr.kind}))
            if dt > OFFSCREEN_S:
                del self.tracks[tr.id]
            return
        if tr.state == OFFSCREEN:  # where it was going is in view again, but it's elsewhere by now: off still
            if dt > OFFSCREEN_S:
                del self.tracks[tr.id]
            return
        if dt >= MISSING_S and tr.state == LIVE:
            if tr.seen < max(2, round(MIN_SEEN_S * HZ)) and tr.solid_at is None:
                self.counts["flickers"] += 1  # seen in a frame or two: a match on scenery, not a monster (one seen
                del self.tracks[tr.id]        # fading in at its spawn point is one: perception.spawns)
                return
            tr.state = HIDDEN
            tr.crowded = tr.crowded or any(
                o.kind == tr.kind and o.id != tr.id and o.state in (LIVE, DYING) and beside(o.mob, tr.mob)
                for o in self.tracks.values())
        due = max(tr.last_seen, self._last_hit(tr)) + self.shape.hide_s
        if tr.state == HIDDEN and t > due:
            if tr.crowded:
                self.counts["lost in a crowd"] += 1
                del self.tracks[tr.id]
                return
            if OTHERS_BY_PLAYER and self.players is not None and near_rival(self.players, tr.x, tr.y) is None:
                self.counts["nobody by it"] += 1  # nobody else on the map could have killed it
                del self.tracks[tr.id]
                return
            self._dies(tr, tr.last_seen, "other", events, t)

    def _resolve(self, t: float, events: list[Event]) -> None:
        """Her EXP gains put on the monsters they were (see the top): bound to the one her attack took, at once, else
        waiting for one to go; while the likeliest of those is still seen, fading, it's `dying`. Her shares too, and
        the deaths seen (`seen_dying`)."""
        for tr in self.tracks.values():
            tr.owed = False
            if tr.state == DYING and tr.gain is None:
                tr.state = LIVE if t - tr.last_seen < MISSING_S else HIDDEN
        self._judge(t, events)
        for key in self.retold:
            self._retold(key, t, events)
        self.retold = []
        left = []
        for g, kind in self.gains:  # the oldest first: each has first pick before the gains after it
            if self.throws_known and t - g <= BIND_S and self._bind(g, kind, t, events):
                continue
            cands = [tr for tr in self.tracks.values() if tr.state in (LIVE, HIDDEN, DYING) and tr.gain is None
                     and tr.died_seen is None and tr.last_seen >= g - self.shape.hide_s and self._aimed(tr, g)
                     and (kind is None or tr.kind == kind)]  # (one seen dying: `_deaths_seen`'s, by its time)
            gone = [tr for tr in cands if tr.state == HIDDEN]
            if t - g <= FADE_S + MISSING_S:
                left.append((g, kind))  # wait for the fade (a newer gain's monster may be an older one's)
                live = [tr for tr in cands if tr.state == LIVE]
                if live:
                    min(live, key=lambda tr: self._likely(tr, g)).state = DYING
                best = min((tr for tr in cands if not tr.owed), key=lambda tr: self._likely(tr, g), default=None)
                if best is not None and best.state == HIDDEN:  # the likeliest is out of sight already: it goes on it
                    best.owed = True
            elif gone:
                self._dies(min(gone, key=lambda tr: self._likely(tr, g)), g, "hers", events, t)
            else:
                self.counts["gains unexplained"] += 1
        self.gains = left
        self.shares = [g for g in self.shares if SHARES and t - g <= BIND_S and not self._bind_share(g, t, events)]
        self._deaths_seen(t, events)  # (after the binding: a death seen this soon is no gain's of this frame)
        self.picked = {g: ids for g, ids in self.picked.items() if t - g <= self.shape.window_s + self.shape.hide_s}
        self.keyed = {g: k for g, k in self.keyed.items() if g in self.picked}

    # --- her kills' monsters, bound (see the top) ------------------------------------------------------------

    def _bind(self, g: float, kind: str | None, t: float, events: list[Event]) -> bool:
        """Gain `g` (of `kind`, None: not known) bound to the monster the attack of the throw keyed GAIN_AFTER_KEY
        before it took (`_picks`), if a throw was and one is there: True."""
        lo, hi = self.shape.gain_after_key
        th = max((th for th in self.throws if lo <= g - th.t <= hi), key=lambda th: th.t, default=None)
        picks = [] if th is None else self._picks(th, kind, g)
        if not picks:
            return False
        self.picked[g] = [tr.id for tr in picks]
        her = self._her_at(th.t + self.shape.pick_s) or th.her
        self.keyed[g] = (th.t, her[0], kind)
        self._victim(picks[0], g, t, events)
        return True

    def _retold(self, key: float, t: float, events: list[Event]) -> None:
        """The throw keyed at `key`, told again the other way (`throw`): a gain bound off it whose monster's death
        isn't seen is let go and bound again, to what the attack took that way (`_bind`); else to one it took there
        whose death was seen since and taken for another's (`_taken_back`: on the recording of 2026-10-26 00:20 her
        victim's death was seen 0.09 s before the frame that told the throw again); else it waits for its monster to
        go as a gain no throw explains."""
        for g, (k, _, kind) in list(self.keyed.items()):
            tr = next((o for o in self.tracks.values() if o.gain == g and not o.share), None)
            if abs(k - key) > RETOLD_S or tr is None or tr.state != DYING or tr.died_seen is not None:
                continue
            tr.gain, tr.state, tr.lands = None, LIVE if t - tr.last_seen < MISSING_S else HIDDEN, None
            self._count("let go")
            events.append(Event("let_go", t, tr.id, {"kind": tr.kind, "gain": round(g, 3), "why": "retold"}))
            del self.picked[g], self.keyed[g]
            if self._bind(g, kind, t, events):
                continue
            th = next((th for th in self.throws if abs(th.t - key) <= RETOLD_S), None)
            back = [] if th is None else self._picks(th, kind, g, dead=True)
            if back:
                self._taken_back(back[0], g, t, events)
            else:
                self.gains.append((g, kind))

    def _taken_back(self, tr: Track, g: float, t: float, events: list[Event]) -> None:
        """Track `tr`, found dead by its death seen (someone else's: no gain of hers was by it then), is gain `g`'s
        kill of hers after all."""
        self.counts["died other"] -= 1
        tr.gain = g
        self._count("kills bound")
        events.append(Event("victim", t, tr.id, {"kind": tr.kind, "x": round(tr.x), "platform": tr.platform,
                                                 "gain": round(g, 3)}))
        self._dies(tr, g, "hers", events, t)

    def _picks(self, th: _Throw, kind: str | None, g: float, dead: bool = False) -> list[Track]:
        """The monsters of `kind` (None: any) throw `th`'s attack can have taken, the likeliest first: as it picked
        (PICK_S after its key: her Shape's pick_s) from where she was then (her place in the frames; else where she was
        when the throw was told), each where its track was at the gain `g`, ahead of her (VICTIM_BEHIND behind her
        middle at most, THROW_REACH ahead: her Shape's ahead) from VICTIM_UP above her feet to VICTIM_DOWN below, those
        seen within FRESH_S of it first, the nearest ahead first. None bound to another gain, nor one seen dying already
        (DEATH_AFTER_GAIN_S: not by this gain's star). With `dead`, only those found dead as someone else's by their
        death seen DEATH_AFTER_GAIN_S after the gain or later (`_retold`)."""
        her = self._her_at(th.t + self.shape.pick_s) or th.her
        if her is None:
            return []
        sign = 1 if th.side == RIGHT else -1
        out = []
        for tr in self.tracks.values():
            if dead:
                if tr.state != DEAD or tr.cause != "other" or tr.share or tr.gain is not None or tr.died_seen is None \
                        or tr.died_seen - g < DEATH_AFTER_GAIN_S:
                    continue
            elif tr.state not in (LIVE, HIDDEN, DYING) or tr.gain is not None or tr.last_seen < g - self.shape.hide_s \
                    or tr.died_seen is not None:
                continue
            if kind is not None and tr.kind != kind:
                continue
            if self.shape.kill is not None:
                out.append(tr)
                continue
            ahead = sign * (self._at(tr, g) - her[0])
            if -VICTIM_BEHIND <= ahead <= self.shape.ahead and -VICTIM_UP <= tr.y - her[1] <= VICTIM_DOWN:
                hurt = kind is not None or not UNKNOWN_HURT_FIRST or tr.bar is not None
                out.append((g - tr.last_seen > FRESH_S, not hurt, abs(ahead), tr.id, tr))
        if self.shape.kill is not None:
            return self._band_picks(th, out, her, sign, kind, g, dead)
        return [a[-1] for a in sorted(out, key=lambda a: a[:4])]

    def _band_picks(self, th: _Throw, tracks: list[Track], her: tuple[float, float], sign: int, kind: str | None,
                    g: float, dead: bool) -> list[Track]:
        """`_picks` where her attack picks by the client's own search (her Shape's `kill`: Magic Claw's), the likeliest
        first: each track where it was at the pick (`_at`, PICK_S after the key: no star flies), its body (the map's
        game data: monsters.bodies) in the first band its search's windows reach from where she was then
        (character.Search.band); those in a band first, by band, the monster her attack choice aimed it at
        (`_Throw.aim`) first among the first band and the next `tie` (the tracker's x error), then those seen within
        FRESH_S of the gain, the nearest; after them the rest the old box holds (VICTIM_BEHIND behind her middle to her
        Shape's `ahead`, its `up` and `down`: a track the tracker has off where the client saw it), the aimed first, the
        nearest. In order: the first band any is in and the next `tie`, the aimed one first, then by band, seen, hurt
        (a gain of no kind), nearest; the bands after, by band, the aimed first; the rest. With `dead` (a death seen
        another's, `_retold`, `_let_go`), only those in a band.

        Against master's nearest-ahead box on the recordings of 2026-11-02 (R1-R3, her 149 labelled kills, labels-v2:
        the replay kit's, gamelab.sim.replay): the first binding right on 143 against 121, of the strong labels 66 of 71
        against 51 (fitted on R1+R2: tie 0, her lands_s; R3 held out: 13 of 15 against 12). README: Her kill's monster
        by Magic Claw's search."""
        kc, s = self.shape.kill, self.shape
        rows = []
        for tr in tracks:
            k, ahead, dy = self._band_of(th, tr, her, sign)
            if k is None and (dead or not (-VICTIM_BEHIND <= ahead <= s.ahead and -s.up <= dy <= s.down)):
                continue
            hurt = kind is not None or not UNKNOWN_HURT_FIRST or tr.bar is not None
            rows.append((k, tr.id == th.aim, g - tr.last_seen <= FRESH_S, hurt, abs(ahead), tr))
        first = min((r[0] for r in rows if r[0] is not None), default=None)

        def order(r):
            k, aimed, fresh, hurt, near, tr = r
            if k is None:
                return (2, not aimed, 0, not hurt, near, tr.id)
            if k <= first + kc.tie:
                return (0, not aimed, k, not fresh, not hurt, near, tr.id)
            return (1, k, not aimed, not fresh, not hurt, near, tr.id)
        return [r[-1] for r in sorted(rows, key=order)]

    def _band_of(self, th: _Throw, tr: Track, her: tuple[float, float], sign: int) -> tuple[int | None, float, float]:
        """Track `tr`'s band in her Shape's search (None: none), and how far ahead of her and below her feet it was, as
        throw `th`'s attack picked (PICK_S after its key) from `her`, facing `sign`."""
        ahead, dy = sign * (self._at(tr, th.t + self.shape.pick_s) - her[0]), tr.y - her[1]
        body = self.bodies.get(tr.kind)
        half = body.half if body is not None else MONSTER_HALF.get(tr.kind, 34.0)
        top = body.top if body is not None else MONSTER_HEIGHT.get(tr.kind, 46.0)
        return self.shape.kill.search.band(ahead, dy, half, top), ahead, dy

    def _her_at(self, t: float) -> tuple[float, float] | None:
        """Where she was in the frame nearest t (within 0.15 s), as placed; None if no frame was."""
        best = min(self._path, key=lambda p: abs(p[0] - t), default=None)
        return None if best is None or abs(best[0] - t) > 0.15 else (best[1], best[2])

    def _victim(self, tr: Track, g: float, t: float, events: list[Event], share: bool = False) -> None:
        """Track `tr` the monster gain (or share) `g` was: dying from now, its id kept as it walks on until her star
        lands on it (`keyed`: its throw's key and where she was; else LANDS_AFTER_GAIN_S after the gain)."""
        tr.state, tr.gain, tr.share = DYING, g, share
        key = self.keyed.get(g)
        if key is None:
            tr.lands = g + LANDS_AFTER_GAIN_S
        elif self.shape.kill is not None:  # (no flight: her hit shows on it its Shape's lands_s after the key)
            tr.lands = key[0] + self.shape.kill.lands_s
        else:
            tr.lands = key[0] + STAR_LANDS_S + abs(tr.x - key[1]) / STAR_FLIGHT
        self._count("shares bound" if share else "kills bound")
        events.append(Event("victim", t, tr.id, {"kind": tr.kind, "x": round(tr.x), "platform": tr.platform,
                                                 "gain": round(g, 3), **({"share": True} if share else {})}))

    def _bind_share(self, g: float, t: float, events: list[Event]) -> bool:
        """Share `g` bound to the monster hurt by her that it was: of the tracks with a bar (a monster's bar shows to
        whoever hurt it) that her throws kept went at, the lowest bar, then the nearest her. True if one was."""
        her_x = self.her[0] if self.her is not None else 0.0
        cands = [tr for tr in self.tracks.values() if tr.state in (LIVE, HIDDEN) and tr.gain is None
                 and tr.bar is not None and any(th.t <= g and tr.id in th.aimed for th in self.throws)]
        if not cands:
            return False
        cands.sort(key=lambda tr: (tr.bar, abs(tr.x - her_x), tr.id))
        self.picked[g] = [tr.id for tr in cands]
        self._victim(cands[0], g, t, events, share=True)
        return True

    def _judge(self, t: float, events: list[Event]) -> None:
        """The tracks bound to a gain or share: dead once their death is seen, or once out of sight MISSING_S from
        CONFIRM_S after it; let go if re-found REBIND_S after it (it was another: `_let_go`)."""
        for tr in list(self.tracks.values()):
            if tr.gain is None or tr.state != DYING:
                continue
            cause = "other" if tr.share else "hers"
            if tr.died_seen is not None:
                self._dies(tr, tr.gain, cause, events, t)
            elif t - tr.gain >= REBIND_S and tr.last_seen >= t - 1e-9:
                self._let_go(tr, t, events)
            elif t - tr.gain >= CONFIRM_S and t - tr.last_seen >= MISSING_S:
                self._dies(tr, tr.gain, cause, events, t)

    def _let_go(self, tr: Track, t: float, events: list[Event]) -> None:
        """Track `tr`, bound to a gain, seen alive since: not its monster. The gain goes on the next its attack took
        that's out of sight now (`picked`), if one is."""
        g, share = tr.gain, tr.share
        tr.gain, tr.share, tr.state, tr.lands = None, False, LIVE, None
        self._count("let go")
        events.append(Event("let_go", t, tr.id, {"kind": tr.kind, "gain": round(g, 3), "why": "seen alive"}))
        if self.shape.kill is not None and not share and BAND_LET_GO_DEAD:  # (a death seen in her search: hers)
            th = self._keyed_throw(g)
            back = [] if th is None else self._picks(th, self.keyed[g][2], g, dead=True)
            if back:
                self._taken_back(back[0], g, t, events)
                return
        nxt = [self.tracks[i] for i in self.picked.get(g, ()) if i in self.tracks and i != tr.id]
        nxt = [o for o in nxt if o.gain is None and o.state in (LIVE, HIDDEN) and t - o.last_seen >= MISSING_S]
        if nxt:
            self._victim(nxt[0], g, t, events, share)
        elif not share:
            self.counts["gains unexplained"] += 1

    def _keyed_throw(self, g: float) -> _Throw | None:
        """The throw gain `g` was bound off (`keyed`), if it's still kept."""
        k = self.keyed.get(g)
        return None if k is None else next((th for th in self.throws if abs(th.t - k[0]) <= RETOLD_S), None)

    def _deaths_seen(self, t: float, events: list[Event]) -> None:
        """The tracks whose death was seen (`seen_dying`) not bound to a gain: one a gain of hers bound to another
        could have been (`picked`), that one not seen dying, takes the gain (its own let go); else, with a gain waiting
        that it can be, that gain; else someone else killed it. A gain only DEATH_AFTER_GAIN_S or more before the death
        was seen: sooner, her star hasn't landed (another player's kill seen dying by her victim isn't hers)."""
        for tr in list(self.tracks.values()):
            if tr.died_seen is None or tr.gain is not None or tr.state == DEAD:
                continue
            self._count("deaths seen")
            other = next((o for g, ids in sorted(self.picked.items(), reverse=True) if tr.id in ids
                          and tr.died_seen - g >= DEATH_AFTER_GAIN_S
                          for o in self.tracks.values() if o.gain == g and o.state == DYING and o.died_seen is None),
                         None)
            if other is not None:
                g, share = other.gain, other.share
                other.gain, other.share, other.state = None, False, LIVE if t - other.last_seen < MISSING_S else HIDDEN
                other.lands = None
                self._count("let go")
                events.append(Event("let_go", t, other.id, {"kind": other.kind, "gain": round(g, 3),
                                                             "why": "another seen dying"}))
                self._victim(tr, g, t, events, share)
                self._dies(tr, g, "other" if share else "hers", events, t)
                continue
            waiting = next(((g, kind) for g, kind in self.gains if (kind is None or kind == tr.kind)
                            and tr.died_seen - g >= DEATH_AFTER_GAIN_S and self._aimed(tr, g)), None)
            if waiting is not None:
                self.gains.remove(waiting)
                self._victim(tr, waiting[0], t, events)
                self._dies(tr, waiting[0], "hers", events, t)
                continue
            self._dies(tr, tr.died_seen, "other", events, t)

    def _count(self, what: str) -> None:
        self.counts[what] = self.counts.get(what, 0) + 1

    def _last_hit(self, tr: Track) -> float:
        """When her last attack its way could have hit it (none: never)."""
        return max((th.t + self.shape.hit_after_s for th in self.throws if tr.id in th.aimed), default=-1e18)

    def _aimed(self, tr: Track, g: float) -> bool:
        if self.throws_known:
            return any(0 <= g - th.t <= self.shape.window_s and tr.id in th.aimed for th in self.throws)
        her, s = self.her, self.shape
        return her is not None and abs(tr.x - her[0]) <= s.ahead and -s.up <= tr.y - her[1] <= s.down

    def _likely(self, tr: Track, g: float) -> tuple:
        """Which of the monsters gain `g` may be comes first (see the top)."""
        went = tr.last_seen >= g - FRAME_SLACK
        faded = went and tr.still_since <= g + FRAME_SLACK
        theirs = PLAYERS_LAST and bool(self.players) and near_rival(self.players, tr.x, tr.y) is not None
        return (not faded, not went, theirs, 1.0 if tr.bar is None else tr.bar,
                abs(tr.x - self.her[0]) if self.her is not None else 0.0)

    def _dies(self, tr: Track, when: float, cause: str, events: list[Event], t: float) -> None:
        tr.state, tr.died_at, tr.cause, tr.judged = DEAD, when, cause, t
        self.counts[f"died {cause}"] += 1
        by = near_rival(self.players, tr.x, tr.y) if cause == "other" and self.players else None
        events.append(Event("died", t, tr.id, {"kind": tr.kind, "cause": cause, "x": round(tr.x),
                                                "platform": tr.platform, "hurt": tr.bar is not None,
                                                "at": round(when, 3), **({"by": by.id} if by is not None else {}),
                                                **({"share": True} if tr.share else {}),
                                                **({"seen": True} if tr.died_seen is not None else {})}))


def overlap(a: Mob, b: Mob) -> bool:
    """Whether a search's sighting can be one re-found: their pictures overlap as suppress_overlaps counts it
    (a search keeps one of two overlapping)."""
    return abs(a.x - b.x) <= max(a.w, b.w) // 2 and abs(a.y - b.y) <= max(a.h, b.h) // 2


def beside(a: Mob, b: Mob) -> bool:
    """Two monsters' pictures touching or overlapping on screen."""
    return abs(a.x - b.x) <= max(a.w, b.w) and abs(a.y - b.y) <= max(a.h, b.h)


def same(a: Mob, b: Mob) -> bool:
    """One detection: the same kind, the same picture at the same place. Two monsters standing close overlap
    on screen, and each track's window finds its own."""
    return (a.kind, a.x, a.y, a.w, a.h) == (b.kind, b.x, b.y, b.w, b.h)


def snapshot(tracks: list[Track]) -> list[Track]:
    """Copies of tracks as they are now (a state keeps them while the tracker goes on)."""
    return [replace(tr) for tr in tracks]
