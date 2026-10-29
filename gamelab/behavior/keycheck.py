"""Her keys doubted, checked and followed to where the game has them: the user's rule of 2026-10-28, after a hunt pressed
a regular attack the game didn't have on its key any more, and their answers of that afternoon: her other keys moved in
the game "Follow them", the binding read "Remember it" (control.keymap), her attack's MISS over a monster "Yes, read it".

What happened (live, 2026-10-28 00:30, Tree Dungeon; D:/MapleHelper-scratch/live/grind-1028-0030.log, telemetry
lab/telemetry/20261028-003049-tour.jsonl): the user had moved the regular attack off Ctrl in their own play and not put
it back before handing over, so the bot's regular attack presses did nothing. At 00:31:14-00:31:33 she pressed it about
20 times at one Horny Mushroom at 28/403 HP ("the regular attack: horny-mushroom ~28/403 HP; finishing it"), its bar
9% throughout; from 00:33:00 regular attacks killed again (the key put back by then). Nothing noticed. The user: "I'd
rather use another way to fix this, 1. Throw 5 times before doubting the key error 2. Check if star has run out 3. Open
keyboard config (\\) to check the key configuration and see if attack button is different, and fix things."

1. **The doubt** (`RegularWatch`, fed by behavior.attacks.Attacks at every throw and look): a regular press is judged at
   the first look JUDGE_FROM_S (attacks.LANDS_S, 0.6 s) or more after its key, JUDGE_BY_S at the latest, and only one
   that should show what it did: at a target (its kind's HP known, not dying), from the ground or a jump (JUMPS_JUDGED:
   as designed a jump's was left out), not on a rope or at its top (the game refuses those), no touch on her
   from KNOCK_S before its key to its look, the target's bar believable before it (between empty and full: hurt, not
   dying) or none up (unhurt: UNHURT_JUDGED; a monster shows its bar once it's hit; a bar a look missed is the one seen
   on it BAR_MEMORY_S before) and the target in her regular attack's reach (`regular_reach`: the star's, REACH_IN
   inside reach.STAR_RANGE, with no other monster nearer that way that her star or claw takes first, the look's or one
   seen lately and hidden, HIDDEN_S; a wand's box) at its key's look and at the judging one. It took **no effect** when that monster is seen again alive with a believable bar as high as before
   (BAR_STEP/2 under it at the most), or still with no bar up if it had none, no kill of hers placed on it and no EXP
   gain naming its kind since SHOWS_S after the key; an **effect** when its bar is lower (or one is up where there was
   none), a kill of hers is placed on it or a gain names its kind (a party share names none: it isn't one). An effect
   a Lucky Seven cast at it keyed after the press could have made (keyed SHOWS_S before the effect or earlier) is the
   cast's: the press is passed over, as every press not judged is (neither counted nor ending a run). A press with her
   attack's MISS over its target (perception.misses: the game's NoRed MISS popup, read off her looks' frames and the
   feed's frame before her next key, behavior.hunter._miss_points), first seen MISS_FROM_S to MISS_BY_S after its key
   and not one the other attack keyed at it could have made (a cast's, for a regular press), took effect (it missed:
   the key works): "missed", the run starts again. DOUBT_PRESSES no-effects in a row are the doubt.
1b. **Her other doubts**, each checked the same way (2.-5.): her Lucky Seven casts judged as her regular presses are (a
   second RegularWatch, `kind` "casts", fed by Attacks; CAST_DOUBT in a row: an arrow moved has her throw the way she
   faced, a Lucky Seven key moved casts nothing); her moves (MoveWatch, told by control.navigator what her minimap dot
   showed of every chain: JUMPS_DOUBT planned jumps that didn't lift her, RUNS_DOUBT runs that didn't move her; the
   check where she is, no retreat on moves doubted, put off DEFERS_MAX times the hunt stops); and out of darts' stop due
   in the hunt loop (behavior.hunter.STARS_CHECKED: her Lucky Seven's key moved takes no MP and kills nothing as no stars
   do), the window read first, the stop raised once it shows her Lucky Seven where she has it (or isn't read), at once
   without a check due (the gap, the game's word); put off, her Lucky Seven held (the regular attack instead) for
   hunter.STARS_WAIT_S at the most. And the hunt loop's stall (behavior.hunt.STUCK_S: no kill, pick-up or new ground
   for 120 s; a "stall" doubt, once a stretch, hunt.STUCK_CHECKS): an arrow or her jump moved can keep her throwing where
   she stands, her chains none; followed, the hunt goes on, else she stands by as before. And her pet food (a "pet"
   doubt, from behavior.pet's: her pet food taps that didn't raise its fullness, Hunter._pet_doubted): PET FOOD
   followed, or on her key as configured, "out of Pet Food?" (her pet's care paged it).
2. **A place to do it**: no monster within PLACE_CLEAR of her in a look, she stays; else away to a safe place as on
   the Lie Detector (behavior.retreat.away: her moves alone, 10 s at most), and a look after it: one still near puts the
   check off (below). A retreat for a check that timed out has this run's later checks wait where she is.
3. **Her stars** (the thief's: her profile's ammo, "throwing stars"; a magician's regular attack is her wand, nothing
   to run out of): a Lucky Seven cast that took MP since the doubt's first press (behavior.darts.DartWatch.mp_shown_at)
   says she has them; else, with the MP for a cast and a step, the Lucky Seven key and a monster in reach with none
   within the claw's reach and CLAW_SLACK of her either way (up close the key swipes, no MP whatever her stars), her on
   a platform, not on a rope nor at its top, untouched, up to STAR_CASTS casts within STAR_TEST_MAX_S: one that takes
   MP (her bar MP_SHOWS_S after its key) or lowers its target's bar (none of its kind within BAR_NEAR of it since: the
   tracks may swap them) says she has them; a touch on her ends the test (not
   known); STAR_CASTS that took no MP and hurt nothing say her stars **may be out**, which isn't the stop: live her casts
   take no MP with her stars in her bag 5-9 in a row (behavior.darts: a monster on her unseen, the key swiping it), 7-8
   in a row with the test's own conditions (the review's count of 2026-10-26), so out of darts stays its own watch's
   (DartWatch: presses_for, 20 on Tree Dungeon, the test's casts counted in its stretch): the test raises OutOfDarts
   only if that's due; else, ATTACK found on her key (5.), no more regular attacks this run. Without the MP it isn't
   known; with no monster to cast at, not known.
4. **The window** (`KeyCheck._window`; perception.keybindings reads it), from a quiet moment (`unquiet`: on a platform,
   not on a rope nor at its top, untouched for KNOCK_S, no monster within PLACE_CLEAR, two looks QUIET_LOOK_S apart
   with her still; one more retreat if a monster's near; none in QUIET_S: put off): a frame first (the client not at
   1366x768: not opened, nothing sent), the mouse moved off where it opens (MOUSE_AT: a pointer over a cell hides it,
   and over a preset page draws that page as "checked"), a frame (open already: read, no backslash), the backslash
   tapped, frames SETTLE_S, AGAIN_S and LATE_S after one another until it's seen open (its whole title, or its name
   alone with the rest off the frame). Not seen: the frame against the one before the backslash where it opens
   (CHANGED_SHARE of the px changed by CHANGE_LEVEL: the live window changed 97% there, 2 s of her moving 70% at the
   most), unchanged it didn't open (no Esc: with nothing open Esc opens the game's MENU, the key bindings have ESC on
   MENU), else it may be open. Seen: its frame saved (lab/shots/keybindings_<time>.png); Esc, frames CLOSE_S and
   CLOSE_AGAIN_S later, Esc once more only if both still show it open, and so once more; still open, it may be open.
   It **may be open** (not seen closed after the backslash, or no frame to tell): the hunt stops (`KeyCheckStop`: her
   retreat, the page "close it with Esc or CANCEL"; what keys do in the game with it open isn't known). Only the
   backslash, Esc and one mouse move are sent; with the Lie Detector alarm up the player refuses them (the alarm goes
   on as ever; a window that may be open then is said and paged).
5. **What then** (`KeyCheck._act`, said in capitals; a telemetry event 'attack_key'). The window read, every key of
   hers is followed to where it shows her function (control.keymap.follow: JUMP, the MOVE arrows, ATTACK, PICK UP, SIT,
   her Lucky Seven's, pet food's and chair's pictures, perception.keybindings.PICTURES); one of her moves, her Lucky
   Seven or her pet food on no key she can press (keymap.NEEDED; never Esc, the backslash or maplehelper's hotkeys, and
   her moves and pet food never on what the Lie Detector's allowance can't take, alarm.ALARM_UNSAFE) stops the hunt
   (KeyCheckStop), nothing followed. Else what moved, or went to no key (her pick-up, chair, sit), is followed
   (Hunter.follow: her Actions' keys, which her chains' key names, the mover, her pet's care and her retreat read as
   they go, and her attack choice's; `keys_read_at`, her keys as read this run), the binding read
   remembered for her next runs (keymap.remember, lab/keybindings-<character>.json) and her moves' and pet food's keys
   handed to the player (keymap.publish, paths.ALARM_KEYS: its allowance with the Lie Detector's alarm up), then:
   - ATTACK on another key: her regular attack is that key from now on (config.toml unchanged), paged; her other keys
     moved said with it ("HER KEYS FOLLOWED: ...");
   - ATTACK on no key (in the tray, or on none), on a key not to press (NOT_ATTACK), or the window not read: no more
     regular attacks this run (Lucky Seven only), paged (remembered so, said at the next start, where the regular
     attack is tried again);
   - ATTACK on her configured key, other keys moved: "followed", said, paged; her Lucky Seven moved, out of darts'
     stretch starts again (DartWatch.followed);
   - nothing moved: a regular doubt's misses were something else ("misses? out of reach?"): no more regular attacks at
     the doubt's targets (`RegularWatch.no_regular_at`, every attack mode's: Hunter.hold_at), the hunt goes on, logged
     only; with her stars maybe out (3.), no more regular attacks this run, paged; her moves' or casts' doubt: "as
     configured", logged only; a pet doubt: "out of Pet Food?"; a stars doubt: out of darts' stop, as it is whatever
     else moved but her Lucky Seven (ATTACK followed or off first).
   At most CHECKS_MAX checks a run and none within CHECK_GAP_S of the last: a doubt in that time holds the regular
   attack at its targets only; once CHECKS_MAX have run, the next doubt turns it off for the run (paged). A doubt that
   gets no check (that, no quiet moment, the window not read, outside the hunt loop) tells her hunter
   (Hunter.keys_doubted): a remembered binding not read this run that has the doubted keys otherwise than config.toml
   is marked (control.keymap.suspect), and the next command starts with config.toml's [keys].

In the hunt loop (`tour --loop`, `--grind`: Hunt.run) the doubt raises KeyDoubt where her health is asked after
(Hunter.health_check), and the hunt loop runs the check (Hunter.check_keys) as it stands still for a look that failed
its checks: its clocks paused over it, the next pass from where she is. Every look of the check asks the safety stop
(the HP watch's due), as health_check would. A check **put off** (a monster near after the retreat, or not quiet in
QUIET_S) holds her regular presses, and health_check raises its doubt again at the first look DEFER_GAP_S or more later
that `unquiet` passes; after DEFERS_MAX of them, no more regular attacks this run (paged). Elsewhere (`hunt kill`,
`hunt clear`, `nav`, a plain `tour`, `probe run`) her regular presses are held (Lucky Seven only) and whoever is at the
PC is paged to run `gamelab keys --read`, once a run for each kind of doubt.

On the live data (gamelab README: The regular attack's key checked, Her keys followed; tests/test_keycheck.py replays the runs of
2026-10-25 to 28 through `RegularWatch`, tests/fixtures/regular_live_2026-10-28.json): at 00:31 the doubt at the look
of 00:31:20.9, its 5th press keyed at 00:31:19.3, 12 s before her last useless press (as built 00:31:19.7); with her
keys right (the other 15 runs, 245 minutes, 595 regular presses) never: 66 judged, 4 without effect, 2 in a row at the
most (as built 60, 5 and 2)."""

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from maplehelper.engine import LEFT, RIGHT

from .. import paths, telemetry
from ..attack import reach
from ..control import alarm, client, keymap
from ..control.navigator import LIFT_BY_S, LIFT_UNITS, RUN_SEEN, RUN_UNITS
from ..localization.monsters import TRACK_SLACK, Monster, seen_again
from ..perception import keybindings
from ..perception.framecheck import BadLook
from .safety import HuntStop

DOUBT_PRESSES = 5      # regular presses in a row that took no effect: the doubt (the user's "5 times")
# ... and her Lucky Seven casts, judged so too (a second RegularWatch, `kind` "casts": the thief's, her star's reach):
# her keys moved in the game (an arrow: she throws the way she faced, a Lucky Seven key that casts nothing) show so. Live
# (the tour runs of 2026-10-25 to 28, keys right, 2,085 casts judged): 89 without effect, 5 in a row at the most
# (D:/MapleHelper-scratch/akey2/moves/casts_replay.py)
CAST_DOUBT = 8
JUDGE_FROM_S = 0.6     # a press judged at the first look this long after its key (attacks.LANDS_S) ...
JUDGE_BY_S = 2.0       # ... and no later (live the first sign of an effect came 0.25-1.27 s after the key, p10-p90,
                       # 1.52 at the most, of 135 regular presses with one)
SHOWS_S = 0.15         # an effect counts from this long after the key (attacks.SHOWS_S)
BAR_STEP = 0.03        # a bar this much lower is her damage: a no-effect's bar is BAR_STEP/2 under the one before at most
KNOCK_S = 0.45         # a touch from this long before the key to the judging look: not judged (hunter.KNOCK_S)
REACH_IN = 50.0        # the target this far inside reach.STAR_RANGE at the key's look and the judging one
# Whether a jump throw's regular press is judged too (by the same rule: its target in her star's reach from where she
# stood). Left out as designed, the doubt never came in 3 of 12 of the simulator's runs with the game's Attack moved
# (each 110-250 regular presses that did nothing, 70-80% of them jump throws at a monster she couldn't finish); the
# live runs judged with them have 4 more effects and no more presses without one (D:/MapleHelper-scratch/akey/build).
JUMPS_JUDGED = True
# Whether a press at a monster with no bar up (unhurt: its bar shows once it's hit) is judged too: a bar up by its look
# an effect, still none no effect. Left out (as built), with the regular attack alone (`--attack regular`, a profile
# without Lucky Seven) and its key wrong no monster is ever hurt, so no press was ever judged: the review's simulator
# runs, 283 and 258 useless presses and no doubt. Live, judged, the runs with her keys right have 10 more presses
# judged (66 against 56) and none more without effect (4), 2 in a row at the most (D:/MapleHelper-scratch/akey/fix).
UNHURT_JUDGED = True
BAR_MEMORY_S = 3.0     # a monster seen with its bar up this recently is hurt still when a look misses its bar (one of
                       # 00:31:14.7's: the Horny Mushroom at 9% read with no bar, between two reads of it at 9%; taken
                       # for unhurt, its bar back at 9% was an effect, and the doubt came 2.4 s later, at 00:31:23.3)
UNHURT_AGAIN_S = 0.3   # an unhurt target's no effect: two looks this far apart with still no bar up (a bar missed once)
BAR_NEAR = 60.0        # her stars' test: another of its target's kind this near it, its bar tells nothing (_crossed)
# Her attack's MISS over a press's target (perception.misses; the user, 2026-10-28, "Yes, read it"): the key works, it
# missed. A press with one over its target first seen MISS_FROM_S to MISS_BY_S after its key (the game draws it where
# her star lands: the star leaves 0.52 s after the key, control.actions.STAR_DELAY_MS, measured at 60 fps, and flies up
# to 0.6 s; the popup reads for about half a second) took effect: `verdict` "missed", the run of no-effects starts
# again. Over it: within MISS_DX across of where its target was, its bottom from MISS_UP over the target's feet to
# MISS_DOWN under them (it's drawn over the monster's head and rises about 30 px a second). A popup is first seen at the
# earliest of its sightings MISS_SEEN_GAP_S apart at the most, MISS_SAME_DX across and MISS_SAME_DY up or down of one
# another; one the other attack (a Lucky Seven cast at it, for her regular presses; a regular press, for her casts)
# keyed MISS_FROM_S to MISS_BY_S before that could have made isn't this press's: it's judged without it, by its
# target's bar. (Until the review of 2026-10-28 MISS_FROM_S was 0.15, a damage's earliest on its target's bar, which
# is the server's, and only a cast keyed after the press was the cast's: a cast's MISS still up 0.25 s after the next
# regular press counted for that press, so a regular key that did nothing went undoubted.)
MISSES_READ = True
MISS_FROM_S = 0.5
MISS_BY_S = 1.6
MISS_SEEN_GAP_S = 0.5
MISS_SAME_DX = 40.0
MISS_SAME_DY = 60.0
MISS_DX = 70.0
MISS_UP = 200.0
MISS_DOWN = 15.0
HIDDEN_S = 1.8         # a monster seen this recently that a look didn't find is about still (fight.HIDDEN_FOR: hit, under
                       # its damage numbers): one nearer a press's target takes its attack (loop_tree_ore_waits)
WALK_S = 120.0         # units a second a target unseen for a while may have walked (a Horny Mushroom's 100 or so)
KEPT_S = 8.0           # s her casts, touches, gains and kills are kept for the presses waiting
STAR_CASTS = 5         # Lucky Seven test casts that took no MP and hurt nothing: her stars may be out (not the stop)
MP_SHOWS_S = 0.55      # a test cast judged by her MP bar this long after its key (attacks.MP_SHOWS_S) ...
STAR_TEST_S = 1.3      # ... and its target's bar, looked at for this long at the most
STAR_TEST_MAX_S = 10.0  # the test's casts within this long (a cast's way waited for while a monster is up close)
CLAW_SLACK = 50.0      # a test cast has no monster within this past the claw's reach (reach.CLAW_FAR) of her, either way
CHECKS_MAX = 3         # checks a run at the most ...
CHECK_GAP_S = 300.0    # ... none within this of the last
SETTLE_S = 0.6         # s after the backslash before the window's frame (her pet's window's: behavior.pet.SETTLE_S) ...
AGAIN_S = 0.4          # ... one more this long after, if it isn't open yet ...
LATE_S = 1.0           # ... and one more this long after that (2 s after the backslash: its opening isn't measured live)
CHANGE_LEVEL = 48      # a px changed (any channel) by more than this ...
CHANGED_SHARE = 0.85   # ... over this share of where the window opens (OPEN_AT): something opened there (the live
                       # window 0.97; 2 s apart in the Lie Detector recordings of her hunting, 0.70 at the most)
CLOSE_S = 0.6          # s after Esc before the frame that says it closed ...
CLOSE_AGAIN_S = 0.4    # ... and one more this long after before Esc again (a window slow to close: no second Esc)
MOUSE_AT = (1100, 300)  # client px at 1366x768 the pointer goes to first: the scene right of where it opens
PLACE_CLEAR = (250.0, 120.0)  # no monster this near (x, y) in a look: she checks where she is (hunter.PET_CLEAR)
QUIET_LOOK_S = 0.3     # her still over two looks this far apart before the backslash ...
QUIET_S = 3.0          # ... within this long (from a retreat's end), else the check is put off
DEFER_GAP_S = 5.0      # a check put off is tried again at a quiet look this long after at the soonest ...
DEFERS_MAX = 3         # ... this many times, then no more regular attacks this run
# Keys the regular attack isn't switched to: Esc (maplehelper's abort, and the game's MENU), the backslash (SET KEY: what
# opens the window), maplehelper's hotkeys (config.toml [hotkeys] and [control]: hidden from the game while it runs): the
# keys none of hers is followed onto (control.keymap.NEVER, with config.toml's)
NOT_ATTACK = keymap.NEVER
KEPT_PRESSES = 12      # a doubt's presses kept for its report (the run's last)
# Her moves doubted (MoveWatch, fed by control.navigator at every chain's end: her jumps that never lifted her and her
# runs that never moved her by her minimap dot on the chain's glances, 15 a second; its LIFT_UNITS and the rest)
# In a row, the doubt: live (the tour runs of 2026-10-24 to 28, 3,408 planned jumps, 1,824 runs of 60 units and more),
# outside the hangs at a rope's top (her dot never moving for 11 and 17 in a row: control.navigator._rope_top since),
# the jumps' that showed no lift 5 in a row at the most (by the telemetry's estimated lift-offs: the chains' plan times
# are exacter), the runs' that didn't move her 3, and a chain cut short for her dot standing where it ran her, 15 of
# 3,393 chains, never 2 in a row (D:/MapleHelper-scratch/akey2/moves)
JUMPS_DOUBT = 6
RUNS_DOUBT = 4
EXIT = 6               # the command's exit code after KeyCheckStop (the safety stop's 4, out of darts' 5)


def _now(stop):
    """A stop held by a stars doubt (out of darts'), raised now: its time (`at`) now, the hunt's end there."""
    if hasattr(stop, "at"):
        stop.at = time.monotonic()
    return stop


def _believable(bar: float | None) -> bool:
    return bar is not None and 0.0 < bar < 1.0


def regular_reach(s, m: Monster) -> bool:
    """Whether her regular attack reaches monster `m` from where look `s` has her: her profile's (character.active):
    a wand's box (Geometry "melee": its near to far ahead, her body's half aside, its down to up), else the star's
    (reach.can_hit, REACH_IN inside reach.STAR_RANGE: walls and floors stop it). Monsters above her as far as the
    search reaches them too: live 8 of 8 presses at one 67-89 above her and 131-269 ahead took effect (the simulator's
    world took a star at none 90 above and 232 ahead, the review's doubt there with her keys right: its own geometry).
    Not with another monster of the look nearer that way that her star or her claw reaches (`_in_front`: the attack
    takes the first, so this one's bar doesn't move with her keys right; loop_tree_ore_waits doubted so, its stars
    taking a Horny Mushroom hidden under its damage numbers: Attacks counts those the watch saw lately in the look,
    RegularWatch.hidden). Live it passes over 6 presses with her keys right, 2 of them without effect."""
    if s is None or s.where is None or s.platform is None:
        return False
    from .. import character
    geo = character.active().attacks["regular"].geometry
    if geo.kind == "melee" and geo.box is not None:
        b, (_, half) = geo.box, reach.body_of(m.kind)
        ahead, dy = abs(m.x - s.where.x), s.where.y - m.y
        return b.near - half <= ahead <= b.far + half and -b.down <= dy <= b.up
    x, y, p = s.where.x, s.where.y, s.platform
    far = reach.STAR_RANGE - REACH_IN
    return reach.can_hit(p, x, y, m, reach=far) and _in_front(s, m, far) is None


def _crossed(m: Monster, monsters) -> bool:
    """Whether another monster of `m`'s kind is within BAR_NEAR across of it (and 60 up or down) in a look: over each
    other the looks' tracks can swap them, and with them their bars."""
    return any(o is not m and o.kind == m.kind and abs(o.x - m.x) < BAR_NEAR and abs(o.y - m.y) < 60 for o in monsters)


def _in_front(s, m: Monster, far: float) -> Monster | None:
    """A monster of look `s` other than `m` on its side of her and nearer, that her star reaches (reach.can_hit) or her
    claw (reach.claw_chance: up close the key swipes it): the attack takes it first. None if there's none."""
    x, y, p = s.where.x, s.where.y, s.platform
    side = 1 if m.x >= x else -1
    for o in getattr(s, "monsters", ()):
        if o is m or (o.kind == m.kind and abs(o.x - m.x) < 1.0 and abs(o.y - m.y) < 1.0):
            continue
        ahead = (o.x - x) * side
        if not -reach.body_of(o.kind)[1] < ahead < abs(m.x - x):
            continue
        top, half = reach.body_of(o.kind)
        if reach.claw_chance(ahead, o.y - y, top, half) > 0 or reach.can_hit(p, x, y, o, reach=far):
            return o
    return None


def near_her(s, model=None) -> list[Monster]:
    """The monsters look `s` has within PLACE_CLEAR of her; on a safe platform (`model`'s: MapModel.safe, none spawn on
    it and none's platform runs under it, and monsters never leave their platform) only those placed on hers, none
    else can touch her there (since 2026-10-28: a Green Mushroom on 24 within 250 of her on 25 had put the check off)."""
    if s is None or s.where is None:
        return []
    near = [m for m in s.monsters if abs(m.x - s.where.x) < PLACE_CLEAR[0] and abs(m.y - s.where.y) < PLACE_CLEAR[1]]
    p = getattr(s, "platform", None)
    if near and model is not None and p is not None and _safe(model, p.id):
        near = [m for m in near if m.platform in (None, p.id)]
    return near


def _safe(model, pid: int) -> bool:
    try:
        return bool(model.safe(pid))
    except Exception:
        return False


def unquiet(h, s) -> str | None:
    """Why look `s` isn't a moment for hunter `h` to open the KEY BINDINGS window in (None: it is): not placed, not on a
    platform (a rope, the air), at a rope's top (out of darts' watch: `at_top`, `roped`), touched within KNOCK_S, or a
    monster within PLACE_CLEAR of her."""
    if s is None or s.where is None:
        return "not placed"
    if s.platform is None or h.on_rope(s):
        return "not on a platform"
    darts = getattr(h, "darts", None)
    if darts is not None and (darts.roped or darts.at_top):
        return "at a rope's top"
    if h.knocked(s):
        return "touched just now"
    if near := near_her(s, getattr(h, "model", None)):
        return f"{', '.join(sorted({m.kind for m in near}))} within {PLACE_CLEAR[0]:.0f} of her"
    return None


def _still(a, b) -> bool:
    """Whether two looks have her where she was: the same platform, within 4 units either way."""
    return (a.platform is not None and b.platform is not None and a.platform.id == b.platform.id
            and abs(a.where.x - b.where.x) <= 4 and abs(a.where.y - b.where.y) <= 4)


@dataclass
class Press:
    """A regular press being judged: its key's time, its target (followed look to look) and that one's bar before it
    (1.0 for one with no bar up: `unhurt`), and what came of it (`verdict`: "effect", "none" or "passed", and why; `at`:
    the look it was judged at, or its effect's time; `after`: the target's bar then, None with still no bar up)."""
    t: float
    m: Monster
    bar: float
    seen: float = 0.0         # when `m` was last seen
    verdict: str | None = None
    why: str = ""
    at: float | None = None
    after: float | None = None
    unhurt: bool = False
    bare: float | None = None  # an unhurt target's first judging look with still no bar up

    @property
    def kind(self) -> str:
        return self.m.kind

    def bar_text(self) -> str:
        b = self.after if self.after is not None else None if self.unhurt else self.bar
        return "no bar" if b is None else f"{round(b * 100)}%"


def _clock(t: float) -> str:
    """A time.monotonic time as the wall clock had it (HH:MM:SS.s)."""
    wall = time.time() - (time.monotonic() - t)
    return time.strftime("%H:%M:%S", time.localtime(wall)) + f".{int((wall % 1) * 10)}"


@dataclass
class Doubt:
    """Her keys doubted: `kind` "regular", DOUBT_PRESSES regular presses in a row that took no effect (RegularWatch.due:
    the presses, the run's last KEPT_PRESSES, when the last was judged, `known`); "moves", her jumps or runs not moving
    her (MoveWatch.due: `what` says how); "stars", out of darts' stop due (`stop`: its OutOfDarts, raised if the window
    shows her Lucky Seven where she has it). The run's how-many-th doubt (`number`)."""
    presses: list[Press]
    known: float
    number: int = 1
    kind: str = "regular"
    what: str = ""
    stop: object = None

    @classmethod
    def moves(cls, what: str, known: float, number: int = 1) -> "Doubt":
        return cls([], known, number, "moves", what)

    @classmethod
    def stall(cls, what: str, known: float) -> "Doubt":
        return cls([], known, 1, "stall", what)

    @classmethod
    def pet(cls, what: str, known: float) -> "Doubt":
        return cls([], known, 1, "pet", what)

    @classmethod
    def stars(cls, stop, known: float) -> "Doubt":
        return cls([], known, 1, "stars", f"OUT OF DARTS DUE ({stop}): the key bindings checked first (her Lucky "
                                          "Seven's key may be another in the game)", stop)

    @property
    def first(self) -> float:
        return self.presses[0].t if self.presses else self.known

    @property
    def targets(self) -> list[Monster]:
        """The monsters its presses went at, each once (as last seen)."""
        out: list[Monster] = []
        for p in self.presses:
            if not any(seen_again(m, [p.m]) is p.m for m in out):
                out.append(p.m)
        return out

    def text(self) -> str:
        if self.kind not in ("regular", "casts"):
            return self.what
        kinds = ", ".join(sorted({p.kind for p in self.presses}))
        bars = "/".join(sorted({p.bar_text() for p in self.presses},
                               key=lambda b: (b == "no bar", float(b[:-1]) if b != "no bar" else 0.0)))
        judged = sorted(p.at - p.t for p in self.presses if p.at is not None)
        each = f", each judged {judged[len(judged) // 2]:.1f} s after its key" if judged else ""
        head = "THE REGULAR ATTACK" if self.kind == "regular" else "HER LUCKY SEVEN"
        return (f"{head} TOOK NO EFFECT {len(self.presses)} TIMES IN A ROW ({kinds}, its bar {bars} "
                f"throughout, {_clock(self.presses[0].t)}-{_clock(self.presses[-1].t)}: in reach, untouched{each})")


@dataclass
class MoveWatch:
    """Her moves doubted (see the module's doc): control.navigator tells it, at each chain's end, of every planned jump
    whose lift-off its glances saw through (`jumped`: whether her dot rose LIFT_UNITS) and every run of RUN_UNITS
    (`ran`: whether her dot moved RUN_SEEN); JUMPS_DOUBT jumps in a row that didn't lift her, or RUNS_DOUBT runs that
    didn't move her, are the doubt (`due`, once; its count starts again). None of it at a rope's top (the navigator
    doesn't tell it then: her dot snapped onto the platform, control.navigator._rope_top)."""
    jumps: int = 0
    runs: int = 0
    counts: Counter = field(default_factory=Counter)
    doubts: int = 0
    _due: Doubt | None = None

    def jumped(self, t: float, lifted: bool) -> None:
        self.counts["lifted" if lifted else "no lift"] += 1
        self.jumps = 0 if lifted else self.jumps + 1
        if self.jumps >= JUMPS_DOUBT and self._due is None:
            self._doubt(t, f"HER JUMPS DIDN'T LIFT HER {self.jumps} TIMES IN A ROW (her minimap dot never rose "
                           f"{LIFT_UNITS:.0f} units within {LIFT_BY_S:g} s of a planned lift-off): her jump key may be "
                           "another in the game")

    def ran(self, t: float, moved: bool) -> None:
        self.counts["ran" if moved else "no run"] += 1
        self.runs = 0 if moved else self.runs + 1
        if self.runs >= RUNS_DOUBT and self._due is None:
            self._doubt(t, f"HER RUNS DIDN'T MOVE HER {self.runs} TIMES IN A ROW (her minimap dot never went "
                           f"{RUN_SEEN:.0f} units in a run of {RUN_UNITS:.0f} and more): an arrow key of hers may be "
                           "another in the game")

    def _doubt(self, t: float, what: str) -> None:
        self.doubts += 1
        self._due = Doubt.moves(what, t, self.doubts)
        self.jumps = self.runs = 0

    def due(self) -> Doubt | None:
        d, self._due = self._due, None
        return d


class KeyDoubt(Exception):
    """Raised by Hunter.health_check in the hunt loop on a doubt (RegularWatch.due), or a check put off coming due
    again: the hunt loop runs the check (Hunter.check_keys) and goes on. Not a HuntStop: the hunt doesn't end."""

    def __init__(self, doubt: Doubt):
        self.doubt = doubt
        super().__init__(doubt.text())


class KeyCheckStop(HuntStop):
    """What the key check can't put right ends the hunt (raised by KeyCheck.run, in the hunt loop): the KEY BINDINGS
    window may be open in the game (not seen closed after the backslash: what her keys do with it open isn't known), or
    the window shows her moves' or her other keys where she can't follow them (her jump or arrows on nothing she can
    press, the game's ATTACK on her Lucky Seven's or pet food's key). As every HuntStop: her retreat, said in capitals,
    paged, the command's own exit code. `window`: whether it's for the window."""
    label = "KEY CHECK"
    title = "Key check"
    event = "key_check_stop"
    exit_code = EXIT
    ends = "to a safe place (no more throws)"
    advice = ("Close the KEY BINDINGS window in the game if it's open (Esc or CANCEL), and put her keys back as config.toml "
              "[keys] has them (or set config.toml) before hunting again.")

    def __init__(self, text: str, window: bool = False, at: float = 0.0):
        self.window, self.at = window, at
        super().__init__(text)

    def page(self) -> tuple[str, str]:
        return (("MapleStory: the key bindings window" if self.window else "MapleStory: her keys"),
                f"gamelab stopped the hunt: {self}.")


@dataclass
class RegularWatch:
    """Her regular presses judged look by look for the doubt (see the module's doc): `pressed`, `cast`, `touched`,
    `heard`, `killed` and `look` feed it; `due` is the doubt, once a run is DOUBT_PRESSES long; `holds` says whether a
    regular press at a target is held. `reach_ok`: whether her regular attack reaches a monster from where a look has
    her (look, monster -> bool: `regular_reach`; None: always)."""
    presses: int = DOUBT_PRESSES
    reach_ok: Callable | None = None
    kind: str = "regular"  # what it judges: her regular presses, or "casts" (her Lucky Seven's: CAST_DOUBT)
    pending: list[Press] = field(default_factory=list)
    run: list[Press] = field(default_factory=list)       # the no-effects in a row so far
    casts: list[tuple[float, Monster]] = field(default_factory=list)  # Lucky Seven keys at a target, the last KEPT_S
    touches: list[float] = field(default_factory=list)   # the looks that saw her touched, the last KEPT_S
    gains: list[tuple[float, frozenset | None]] = field(default_factory=list)  # her EXP's gains: (frame, kinds)
    kills: list[tuple] = field(default_factory=list)     # her kills placed: (when, kind, platform, x, track)
    bars: list[tuple] = field(default_factory=list)      # monsters seen with a bar up, BAR_MEMORY_S: (t, monster)
    recent: list[tuple] = field(default_factory=list)    # every monster seen, HIDDEN_S: (t, monster)
    misses: list[tuple] = field(default_factory=list)    # her attack's MISS popups seen, KEPT_S: (t, map x, map y)
    counts: Counter = field(default_factory=Counter)     # presses by what came of them (and why passed over)
    longest: int = 0                                     # the longest run of no-effects
    held: bool = False                                   # regular presses held (a doubt waiting for its check)
    skill_held: bool = False                             # Lucky Seven held (out of darts due, its check put off: the
                                                         # regular attack instead; KeyCheck._defer)
    off: str | None = None                               # why the regular attack is off for the run (None: it isn't)
    no_regular_at: list[Monster] = field(default_factory=list)  # targets it skips (a check found the key right)
    doubts: list[Doubt] = field(default_factory=list)
    checks: list[float] = field(default_factory=list)    # when each check ran
    deferred: Doubt | None = None                        # a check put off (KeyCheck: not quiet) ...
    deferred_at: float = 0.0                             # ... when ...
    defers: int = 0                                      # ... and how many times since the last check
    _due: Doubt | None = None

    # --- fed ------------------------------------------------------------------------------------------------------

    def pressed(self, t: float, m: Monster | None, *, known: bool = True, dying: bool = False, ground: bool = True,
                roped: bool = False, reach_ok: bool = True) -> None:
        """A regular press keyed at `t` at target `m` (None: nothing seen in its way), with what decides whether it's
        judged: its kind's HP `known`, it not `dying`, from the `ground`, her not `roped` (on a rope or at its top),
        her reach of it at the key's look (`reach_ok`)."""
        bar = None if m is None else m.bar if m.bar is not None else self._bar_seen(m, t)
        unhurt = m is not None and bar is None and UNHURT_JUDGED
        why = ("nothing in its way" if m is None else "its HP not known" if not known else "dying" if dying else
               "a jump's" if not ground and not JUMPS_JUDGED else "on a rope" if roped else
               "no bar to judge by" if not _believable(bar) and not unhurt else "out of reach" if not reach_ok else "")
        if why:
            self.counts[f"passed: {why}"] += 1
            return
        self.pending.append(Press(t, m, 1.0 if unhurt else float(bar), t, unhurt=unhurt))

    def _bar_seen(self, m: Monster, t: float) -> float | None:
        """The bar a monster seen with none up had in a look BAR_MEMORY_S or less before `t` (its track, or of its
        kind on its platform where it can have walked since), or None: unhurt as far as her looks know."""
        seen = [(bt, o) for bt, o in self.bars if t - bt <= BAR_MEMORY_S and o.kind == m.kind
                and ((track := getattr(m, "track", None)) is not None and getattr(o, "track", None) == track
                     or o.platform == m.platform and abs(o.x - m.x) < TRACK_SLACK + WALK_S * (t - bt))]
        return float(max(seen, key=lambda x: x[0])[1].bar) if seen else None

    def hidden(self, t: float, monsters: list[Monster]) -> list[Monster]:
        """The monsters looks saw within HIDDEN_S before `t` that `monsters` (a look's) doesn't have: about still,
        under their damage numbers (each once, as last seen)."""
        out: list[Monster] = []
        for seen_at, m in sorted(self.recent, key=lambda r: -r[0]):
            if t - seen_at <= HIDDEN_S and seen_again(m, monsters) is None and seen_again(m, out) is None:
                out.append(m)
        return out

    def cast(self, t: float, m: Monster | None) -> None:
        """A Lucky Seven cast keyed at `t` at `m`: an effect after it could be its."""
        if m is not None:
            self.casts.append((t, m))

    def touched(self, t: float) -> None:
        """A look at `t` saw her HP drop (a touch)."""
        self.touches.append(t)

    def heard(self, gains: list[tuple[float, frozenset | None]]) -> None:
        """Her EXP's gains (each frame's time and the kinds it names: None, any, which no press takes for its own)."""
        self.gains.extend(gains)

    def killed(self, t: float, kind: str | None, platform: int | None, x: float, track: int | None = None) -> None:
        """A kill of hers placed (behavior.ledger.Kill: its gain's time, kind, platform, x, track)."""
        if kind is not None:
            self.kills.append((t, kind, platform, x, track))

    def look(self, t: float, monsters: list[Monster], s=None) -> None:
        """A look at `t` that searched for monsters (`s`: its Scene, for `reach_ok`): the presses waiting judged, the
        run kept in their keys' order (a press is counted once those before it are)."""
        horizon = t - KEPT_S
        self.casts = [c for c in self.casts if c[0] > horizon]
        self.touches = [h for h in self.touches if h > horizon]
        self.gains = [g for g in self.gains if g[0] > horizon]
        self.kills = [k for k in self.kills if k[0] > horizon]
        self.bars = [b for b in self.bars if b[0] > t - BAR_MEMORY_S] + [(t, m) for m in monsters if _believable(m.bar)]
        self.recent = [r for r in self.recent if r[0] > t - HIDDEN_S] + [(t, m) for m in monsters]
        self.misses = [x for x in self.misses if x[0] > horizon] + [(t, mx, my) for mx, my in
                                                                    (getattr(s, "misses", None) or ())]
        # (and those behavior.hunter saw before her keys since, off the feed's newest state: Hunter._miss_glance)
        for p in self.pending:
            if p.verdict is None:
                self._judge(p, t, monsters, s)
        while self.pending and self.pending[0].verdict is not None:
            p = self.pending.pop(0)
            self.counts[p.verdict if p.verdict != "passed" else f"passed: {p.why}"] += 1
            if p.verdict in ("effect", "missed"):  # (a MISS over it: the key works)
                self.run = []
            elif p.verdict == "none":
                self.run.append(p)
                self.longest = max(self.longest, len(self.run))
                if len(self.run) >= self.presses and self._due is None and not self.held and self.off is None:
                    self._due = Doubt(list(self.run[-KEPT_PRESSES:]), t, len(self.doubts) + 1, kind=self.kind)
        self.no_regular_at = [seen_again(m, monsters) or m for m in self.no_regular_at]

    def _judge(self, p: Press, t: float, monsters: list[Monster], s) -> None:
        if any(p.t - KNOCK_S <= h <= t for h in self.touches):
            p.verdict, p.why = "passed", "touched"
            return
        again = self._follow(p, t, monsters)
        effect = self._effect(p, t, again)
        if effect is not None:
            if any(p.t < c_t <= effect - SHOWS_S and seen_again(c_m, [p.m]) is p.m for c_t, c_m in self.casts):
                p.verdict, p.why, p.at = "passed", "a cast's effect maybe", effect
            else:
                p.verdict, p.at = "effect", effect
            return
        if again is not None:
            p.m, p.seen = again, t
        if (missed := self._missed(p, t)) is not None:
            p.verdict, p.at = "missed", missed
            return
        if t - p.t < JUDGE_FROM_S:
            return
        if t - p.t > JUDGE_BY_S:
            p.verdict, p.why = "passed", "not judged in time"
            return
        if again is None:
            return
        if p.unhurt:
            if again.bar is not None:  # (a bar up not lower than full, or none to go by: not told yet)
                return
            if p.bare is None or t - p.bare < UNHURT_AGAIN_S:  # (once more: a bar missed once isn't none)
                p.bare = p.bare if p.bare is not None else t
                return
        elif not _believable(again.bar) or again.bar > p.bar + BAR_STEP:
            return  # (no bar to go by, or one higher than its own was: another monster's)
        if self.reach_ok is not None and s is not None and not self.reach_ok(s, again):
            p.verdict, p.why = "passed", "out of reach at its look"
            return
        p.verdict, p.at, p.after = "none", t, None if again.bar is None else float(again.bar)

    def _follow(self, p: Press, t: float, monsters: list[Monster]) -> Monster | None:
        """Press `p`'s target in a look at `t`: seen again (localization.monsters.seen_again: its track, or of its kind
        near where it was), else the one monster of its kind on its platform it can have walked to since it was last
        seen (TRACK_SLACK and WALK_S a second: a look without the feed's tracks, or one between that missed it)."""
        if (again := seen_again(p.m, monsters)) is not None:
            return again
        far = TRACK_SLACK + WALK_S * max(0.0, t - p.seen)
        near = [m for m in monsters if m.kind == p.kind and m.platform == p.m.platform and abs(m.x - p.m.x) < far
                and getattr(m, "track", None) is None]
        return near[0] if len(near) == 1 else None

    def _missed(self, p: Press, t: float) -> float | None:
        """When a MISS popup over press `p`'s target was seen (by `t`) that's this press's: first seen its key's
        MISS_FROM_S to MISS_BY_S after (`_first_seen`), and not one the other attack keyed at it could have made (see
        MISS_FROM_S); None."""
        for mt, mx, my in sorted(self.misses):
            if mt > t or not (abs(mx - p.m.x) <= MISS_DX and p.m.y - MISS_UP <= my <= p.m.y + MISS_DOWN):
                continue
            first = self._first_seen(mt, mx, my)
            if not p.t + MISS_FROM_S <= first <= p.t + MISS_BY_S:
                continue
            if any(c_t + MISS_FROM_S <= first <= c_t + MISS_BY_S and seen_again(c_m, [p.m]) is p.m
                   for c_t, c_m in self.casts):
                continue
            return mt
        return None

    def _first_seen(self, mt: float, mx: float, my: float) -> float:
        """When the MISS popup seen at `mt` at map (mx, my) was first seen: its sightings before it MISS_SEEN_GAP_S
        apart at the most, near where it was (MISS_SAME_DX, MISS_SAME_DY)."""
        first = mt
        for ot, ox, oy in sorted(self.misses, reverse=True):
            if ot < first and first - ot <= MISS_SEEN_GAP_S and abs(ox - mx) < MISS_SAME_DX and abs(oy - my) < MISS_SAME_DY:
                first = ot
        return first

    def boxes(self, t: float) -> list[Monster]:
        """The targets of the presses waiting whose MISS a look at `t` could show (MISS_FROM_S to MISS_BY_S after their
        keys): where behavior.hunter looks for her attack's MISS popups."""
        return [p.m for p in self.pending if p.verdict is None and p.t + MISS_FROM_S <= t <= p.t + MISS_BY_S]

    def _effect(self, p: Press, t: float, again: Monster | None) -> float | None:
        """When an effect of press `p` (or of a cast after it) shows by `t`: its target's bar lower (one up where none
        was: under full), a kill of hers placed on it, a gain naming its kind; None."""
        times = []
        if again is not None and _believable(again.bar) and again.bar < p.bar - BAR_STEP / 2 and t >= p.t + SHOWS_S:
            times.append(t)
        times += [g for g, kinds in self.gains if p.t + SHOWS_S <= g <= t and kinds is not None and p.kind in kinds]
        for k_t, kind, platform, x, track in self.kills:
            mine = (track == p.m.track if track is not None and getattr(p.m, "track", None) is not None
                    else platform == p.m.platform and abs(x - p.m.x) < TRACK_SLACK)
            if p.t + SHOWS_S <= k_t and kind == p.kind and mine:
                times.append(k_t)
        return min(times) if times else None

    # --- asked ----------------------------------------------------------------------------------------------------

    def due(self) -> Doubt | None:
        """The doubt, once (DOUBT_PRESSES no-effects in a row), else None; its run starts again."""
        d, self._due = self._due, None
        if d is not None:
            self.doubts.append(d)
            self.run = []
        return d

    def again(self, t: float) -> Doubt | None:
        """A check put off whose doubt is due again at `t` (DEFER_GAP_S since): taken (the caller raises it), else
        None. The caller asks only at a quiet look (`unquiet`)."""
        if self.deferred is None or t - self.deferred_at < DEFER_GAP_S or self.off is not None:
            return None
        d, self.deferred = self.deferred, None
        return d

    def holds(self, m: Monster | None) -> bool:
        """Whether a regular press at `m` is held: a doubt waiting for its check, the regular attack off for the run,
        or `m` one of the targets a check found the key right at."""
        if self.held or self.off is not None:
            return True
        return m is not None and any(seen_again(h, [m]) is m for h in self.no_regular_at)

    def checked(self, t: float) -> None:
        """A check ran at `t` (the window opened): the presses waiting and the run start again."""
        self.checks.append(t)
        self.run, self.pending, self.deferred, self.defers = [], [], None, 0

    def may_check(self, t: float) -> str | None:
        """Why a doubt at `t` gets no check (None: it does): CHECKS_MAX done, or the last within CHECK_GAP_S."""
        if len(self.checks) >= CHECKS_MAX:
            return f"{CHECKS_MAX} checks this run already"
        if self.checks and t - self.checks[-1] < CHECK_GAP_S:
            return f"the last check {t - self.checks[-1]:.0f} s ago (none within {CHECK_GAP_S:.0f} s)"
        return None

    def summary(self) -> str:
        """For the end of a run's log: her regular presses judged and the longest run without effect, once a doubt
        came ("" else: a run's log the same as before the watch)."""
        c = self.counts
        if not self.doubts:
            return ""
        out = (f"her regular presses: {c['effect'] + c['none'] + c['missed']} judged, {c['none']} took no effect, "
               + (f"{c['missed']} MISSed (her attack's MISS over its target), " if c["missed"] else "")
               + f"{self.longest} in a row at the most (the doubt at {self.presses})")
        out += f"; {len(self.doubts)} doubt(s), {len(self.checks)} check(s)"
        if self.deferred is not None:
            out += f"; a check put off {self.defers} time(s), not run (her regular presses held)"
        if self.off:
            out += f"; the regular attack off: {self.off}"
        return out


# --- the check ----------------------------------------------------------------------------------------------------

@dataclass
class Outcome:
    """What a check found and did: `what` "switched", "off", "as configured", "held" (no check: CHECKS_MAX, the gap) or
    "put off" (not quiet: tried again later), the key the game has ATTACK on (`key`), the window's reading, what was
    said in capitals, how her stars were told (`stars`: "shown", a cast's MP since the doubt's first press; "test", a
    test cast took MP or hurt its target; "maybe out", STAR_CASTS test casts didn't; "not known", "no target" (no
    monster to cast at); "no stars", a magician), and how long it took."""
    what: str
    text: str
    key: str | None = None
    reading: keybindings.Reading | None = None
    stars: str = ""
    seconds: float = 0.0


class KeyCheck:
    """The check after a doubt (see the module's doc) by hunter `h`: its senses, actions, attack choice, out of darts'
    watch and its log. `reader`: the window's (perception.keybindings.Reader; made when first needed). The simulator's
    eyes draw the window from the world's bindings (sim.fakes.FakeEyes._capture). `window_state` after `_window`:
    "not opened" (nothing sent, or the backslash and nothing changed), "closed" (seen open, then closed), "may be open";
    `sent`: whether the backslash went out."""

    def __init__(self, h, reader: keybindings.Reader | None = None, log: Callable[[str], None] | None = None):
        self.h = h
        self.log = log or h.log
        self._reader = reader
        self.frames_kept = 0
        self._frame_read = None  # the frame the window was last read on
        self._last = None        # the last frame taken
        self.sent = False
        self.window_state = "not opened"
        self.kept = None  # the window's frame as saved (lab/shots), the last read's
        self.no_retreat = False  # a retreat for a check timed out: later checks wait where she is (her moves doubted)

    @property
    def reader(self) -> keybindings.Reader:
        if self._reader is None:
            self._reader = keybindings.Reader()
        return self._reader

    def run(self, doubt: Doubt) -> Outcome:
        """The check: a place, her stars (a regular doubt's), a quiet moment, the window, what then (each said). Raises
        OutOfDarts (out of darts' own watch due after her stars' test; a stars doubt's stop, the window showing her Lucky
        Seven where she has it), SafetyStop (the HP watch, at every look), KeyCheckStop (the window may be open; her keys
        where she can't follow them) and client.LieDetectorAlarm (the player refuses its keys then)."""
        start = time.monotonic()
        watch = self.h.attacks.watch
        if (why := watch.may_check(start)) is not None:
            return self._hold(doubt, why)
        if doubt.kind == "regular":
            self.h.hold_regular()  # (held through it: what it finds lets go)
        stars = ""
        moves_ok = doubt.kind != "moves" and not self.no_retreat  # (her moves doubted, or failed one: no retreat)
        if (why := self._place(retreat=moves_ok)) is None:
            if doubt.kind == "regular":
                stars = self._stars(doubt)
            why = self._quiet(retreat=moves_ok)
        if why is not None:
            out = self._defer(doubt, why)
        else:
            watch.checked(time.monotonic())
            watch.skill_held = False  # (what it finds lets go, or stops the hunt)
            if (casts := getattr(self.h.attacks, "casts", None)) is not None:
                casts.run, casts.pending = [], []
            reading = self._window()
            if self.window_state == "may be open":
                text = ("the KEY BINDINGS window may be open in the game (the backslash sent, it wasn't seen closed): "
                        "close it with Esc or CANCEL" + ("" if reading is None or not reading.read
                                                          else f"; it read {reading.text()}"))
                telemetry.emit("event", kind="attack_key", text=text, what="stopped", stars=stars, doubt=doubt.kind,
                               seconds=round(time.monotonic() - start, 2),
                               **({"reading": reading.record()} if reading is not None else {}))
                raise KeyCheckStop(text, window=True, at=time.monotonic())
            out = self._act(doubt, reading, stars)
        out.stars, out.seconds = stars, time.monotonic() - start
        telemetry.emit("event", kind="attack_key", text=out.text, what=out.what, key=out.key, stars=stars,
                       doubt=doubt.kind, seconds=round(out.seconds, 2),
                       **({"reading": out.reading.record()} if out.reading is not None else {}))
        return out

    def _look(self):
        """A look of the check's: the safety stop asked of it as Hunter.health_check would (her HP watch's due)."""
        s = self.h.look()
        hp = getattr(self.h, "hp_watch", None)
        if hp is not None and (stop := hp.due()) is not None:
            raise stop
        return s

    # --- 1. a place ------------------------------------------------------------------------------------------------

    def _place(self, retreat: bool = True) -> str | None:
        """No monster within PLACE_CLEAR of her in a look: she stays; else away to a safe place (behavior.retreat) and
        a look after it (not with `retreat` False: her moves doubted, the check waits for a quiet moment where she is).
        Why the check is put off (one still near), or None."""
        from .retreat import KEY_CHECK, away
        model = getattr(self.h, "model", None)
        near = near_her(self._look(), model)
        if not near:
            self.log(f"  the key check where she is: no monster within {PLACE_CLEAR[0]:.0f} of her")
            return None
        if not retreat:
            return f"{', '.join(sorted({m.kind for m in near}))} near her (her moves doubted: no retreat)"
        self.log(f"  the key check: {', '.join(m.kind for m in near)} near her: to a safe place first")
        r = away(self.h, time.monotonic(), KEY_CHECK, self.log)
        self._retreated(r)
        if near := near_her(self._look(), model):
            return (f"{', '.join(sorted({m.kind for m in near}))} near her still after the retreat "
                    f"({getattr(r, 'result', '?')})")
        return None

    # --- 2. her stars ----------------------------------------------------------------------------------------------

    def _stars(self, doubt: Doubt) -> str:
        """Whether she has her stars (the thief's: see the module's doc). Returns what told it; raises OutOfDarts only
        when out of darts' own watch is due after the test's casts."""
        from .. import character
        from ..attack import choice
        from .darts import LOW_STEPS
        if character.active().ammo != "throwing stars":
            self.log("  her stars: none to run out of (her regular attack isn't a throw)")
            return "no stars"
        darts, a = self.h.darts, self.h.attacks
        if (shown := darts.mp_shown_at) is not None and shown >= doubt.first:
            self.log(f"  her stars: a Lucky Seven cast took MP {time.monotonic() - shown:.1f} s ago, since the doubt's "
                     "first press: she has them")
            return "shown"
        if a.skill is None or self.h.actions.keys.lucky_seven is None:
            self.log("  her stars: not known (no Lucky Seven key)")
            return "not known"
        took = casts = 0
        t0 = time.monotonic()
        end = t0 + STAR_TEST_MAX_S
        waited = ""
        while casts < STAR_CASTS and time.monotonic() < end:
            s = self._look()
            if (hit := self.h.hit_at) is not None and hit > t0:
                self.log(f"  her stars: not known (touched during the test, after {casts} cast(s))")
                return "not known"
            mp = a.mp
            if mp is None or not choice.can_cast(mp, a.max_mp, a.skill.mp + a.max_mp / choice.MP_STEPS):
                have = "not read" if mp is None else f"{max(0.0, choice.mp_points(mp, a.max_mp)):.0f}"
                self.log(f"  her stars: not known (MP {have} < a cast's {a.skill.mp:.0f} and a step)")
                return "not known"
            aim = self._cast_side(s, doubt)
            if isinstance(aim, str):
                if not casts and aim == "no monster in reach":
                    self.log("  her stars: not known (no monster in reach to cast at, out of the claw's)")
                    return "no target"
                waited = aim
                time.sleep(0.2)  # (the monsters walk, a knock ends: another look)
                continue
            side, target = aim
            if not self._cast(side):
                time.sleep(0.2)
                continue
            casts += 1
            key = self.h.actions.attack_at
            if (shown := self._evidence(key, target)) is not None:
                self.log(f"  her stars: the test's cast {casts} {shown}: she has them")
                return "test"
            if (hit := self.h.hit_at) is not None and hit > t0:
                self.log(f"  her stars: not known (touched during the test, after {casts} cast(s))")
                return "not known"
            took += 1
            if darts.share * choice.MP_STEPS < LOW_STEPS:
                self.log("  her stars: not known (her MP too low to tell)")
                return "not known"
        if took < STAR_CASTS:
            self.log(f"  her stars: not known ({casts} test casts in {STAR_TEST_MAX_S:.0f} s, {took} of them took no MP; "
                     f"{STAR_CASTS} needed" + (f"; waited: {waited}" if waited else "") + ")")
            return "not known"
        if (stop := darts.due(time.monotonic())) is not None:  # (out of darts' own stretch long enough with them)
            if stop.said is None and hasattr(self.h, "_messages"):
                stop.heard(self.h._messages())
            raise stop
        self.log(f"  her stars: MAYBE OUT ({took} test casts took no MP and hurt nothing; with her stars a monster on "
                 f"her unseen does that too): out of darts' own watch stops the hunt at {darts.presses} Lucky Seven "
                 "presses in a row without MP")
        return "maybe out"

    def _cast_side(self, s, doubt: Doubt):
        """Which way a test cast goes, and at what: a monster in reach from where she is in look `s`, none within the
        claw's reach and CLAW_SLACK of her either way (up close the key swipes, no MP), she on a platform, not on a rope
        nor at its top, untouched; the doubt's last target's way first. Else why not (a str)."""
        a, darts = self.h.attacks, self.h.darts
        if (s is None or s.where is None or s.platform is None or self.h.on_rope(s) or darts.roped
                or darts.at_top):
            return "not on a platform"
        if self.h.knocked(s):
            return "touched just now"
        x, y = s.where.x, s.where.y
        claw = reach.CLAW_FAR + CLAW_SLACK
        if any(abs(m.x - x) < claw and abs(m.y - y) < 60 for m in s.monsters):
            return "a monster up close"
        first = RIGHT if doubt.presses[-1].m.x > x else LEFT
        for side in (first, LEFT if first == RIGHT else RIGHT):
            m = a.target(side)
            if m is not None and abs(m.x - x) >= claw and reach.can_hit(s.platform, x, y, m):
                return side, m
        return "no monster in reach"

    def _cast(self, side: str) -> bool:
        """One Lucky Seven press that way, forced (Attacks.forced: chosen, logged and its MP reckoned as any cast)."""
        a, act = self.h.attacks, self.h.actions
        saved = act.skill, act.choose
        act.skill, act.choose, a.forced = True, a.key, "lucky_seven"
        try:
            return act.throw(side) is not False
        finally:
            act.skill, act.choose, a.forced = saved[0], saved[1], None

    def _evidence(self, key: float, target: Monster) -> str | None:
        """Looks after a test cast keyed at `key` at `target`, STAR_TEST_S at the most: what showed it took effect (its
        MP on her bar, DartWatch.mp_shown_at after the key; its target's bar lower, or one up where none was), or None
        (her bar read MP_SHOWS_S after it without, and the time up). Its target's bar tells nothing once another monster
        of its kind has come within BAR_NEAR of it since the key (`_crossed`: the looks' tracks may have swapped them):
        found after the rebase onto her touches (2026-10-28, loop_tree_attack_no_darts: a Horny Mushroom at 35% taken
        for the one at 3.5% walking over it, "lowered", "she has them", no star left)."""
        darts, before, crossed = self.h.darts, target.bar, False
        while time.monotonic() < key + STAR_TEST_S:
            s = self._look()
            if darts.mp_shown_at is not None and darts.mp_shown_at > key:
                return "took MP"
            again = seen_again(target, s.monsters) if s is not None else None
            crossed = crossed or (again is not None and _crossed(again, s.monsters))
            if (again is not None and _believable(again.bar) and s.t >= key + SHOWS_S and not crossed
                    and (not _believable(before) or again.bar < before - BAR_STEP / 2)):
                return "lowered its target's bar"
        return None

    # --- 3. a quiet moment -----------------------------------------------------------------------------------------

    def _quiet(self, retreat: bool = True) -> str | None:
        """A quiet moment for the window (`unquiet`, her still over two looks QUIET_LOOK_S apart) within QUIET_S, one
        more retreat if a monster's near (`retreat`). Why the check is put off, or None."""
        from .retreat import KEY_CHECK, away
        retreated, last, why = not retreat, None, "not looked"
        end = time.monotonic() + QUIET_S
        while True:
            s = self._look()
            why = unquiet(self.h, s)
            if why is None and last is not None and _still(last, s):
                return None
            if why is not None and near_her(s, getattr(self.h, "model", None)) and not retreated:
                self.log(f"  the key check: {why}: to a safe place first")
                self._retreated(away(self.h, time.monotonic(), KEY_CHECK, self.log))
                retreated, last, end = True, None, time.monotonic() + QUIET_S
                continue
            if time.monotonic() >= end:
                return why or "not standing still"
            last = s if why is None else None
            time.sleep(QUIET_LOOK_S)

    def _retreated(self, r) -> None:
        """A retreat for the check that timed out (her trip there not over in its time: an arrow or her jump on another
        key in the game can do that; the review of 2026-10-28, a casts doubt put off twice so): later checks this run
        wait for a quiet moment where she is, as for her moves doubted."""
        if getattr(r, "result", None) == "timeout" and not self.no_retreat:
            self.no_retreat = True
            self.log("  her retreat for the key check timed out (her moves' keys may be others in the game): later "
                     "checks wait for a quiet moment where she is")

    def _unchecked(self, doubt: Doubt, why: str) -> None:
        """A doubt that gets no check (`why`): her hunter told (Hunter.keys_doubted: a remembered binding not read
        this run is marked so, and the next command starts with config.toml's [keys])."""
        if (told := getattr(self.h, "keys_doubted", None)) is not None:
            told(doubt.kind, why)

    def _defer(self, doubt: Doubt, why: str) -> Outcome:
        """The check put off (`why`): its doubt raised again at a quiet look (RegularWatch.again), a regular doubt's
        presses held; past DEFERS_MAX, a regular doubt's: no more regular attacks this run; a stars doubt's: its out of
        darts stop; her moves': the hunt stops (KeyCheckStop)."""
        w = self.h.attacks.watch
        w.defers += 1
        if w.defers > DEFERS_MAX:
            w.deferred = None
            self._unchecked(doubt, f"no quiet moment for the key check ({why})")
            if doubt.kind == "stars":
                self.log(f"KEY CHECK: no quiet moment for it ({DEFERS_MAX} times put off; {why}): out of darts' stop")
                raise _now(doubt.stop)
            if doubt.kind == "moves":
                text = (f"her moves doubted ({doubt.what.split(' (')[0].lower()}) and no quiet moment to check her key "
                        f"bindings ({DEFERS_MAX} times put off; {why})")
                self.log(f"KEY CHECK: {text.upper()}")
                raise KeyCheckStop(text, at=time.monotonic())
            if doubt.kind in ("casts", "stall", "pet"):
                self.log(f"HER KEYS: no quiet moment for the key check ({DEFERS_MAX} times put off; {why}): nothing done")
                return Outcome("put off", "no quiet moment")
            return self._off(f"ATTACK KEY: NO QUIET MOMENT FOR THE KEY CHECK ({DEFERS_MAX} TIMES PUT OFF; {why}): no more "
                             "regular attacks this run (Lucky Seven only)", f"no quiet moment for the key check ({why})")
        w.deferred, w.deferred_at = doubt, time.monotonic()
        if doubt.kind == "stars":  # (her casts on a key that may do nothing: the regular attack meanwhile)
            w.skill_held = True
        held = (": her regular presses held" if doubt.kind == "regular" else
                ": her Lucky Seven held (the regular attack instead)" if doubt.kind == "stars" else "")
        text = (f"{'ATTACK KEY' if doubt.kind == 'regular' else 'HER KEYS'}: the key check put off ({why}){held}; again "
                f"at a quiet moment ({w.defers} of {DEFERS_MAX})")
        self.log(text)
        return Outcome("put off", text)

    # --- 4. the window ---------------------------------------------------------------------------------------------

    def _window(self) -> keybindings.Reading | None:
        """The KEY BINDINGS window opened, read and closed (see the module's doc): its reading (None: no frame);
        `window_state` and `sent` say how it was left."""
        client.check_alarm()
        act = self.h.actions
        self.sent, self.window_state = False, "not opened"
        try:
            before = self._frame()
            if before is None:
                self.log("  the key bindings: no frame of the game to read them on: not opened")
                return None
            fh, fw = before.shape[:2]
            if (fw, fh) != keybindings.FRAME:
                self.log(f"  the key bindings: the client is {fw}x{fh}, not {keybindings.FRAME[0]}x"
                         f"{keybindings.FRAME[1]}: not opened (the window is read at that size)")
                return keybindings.Reading(False, f"the frame is {fw}x{fh}, not {keybindings.FRAME[0]}x"
                                                  f"{keybindings.FRAME[1]}")
            act.play([[0, "mouse", *MOUSE_AT]], "key_bindings_mouse")
            reading = self._read()
            if reading is None:
                self.log("  the key bindings: no frame after the mouse moved: not opened")
                return None
            already = reading.open
            if already:
                self.window_state = "may be open"
            else:
                behind = self._last
                act.play([[0, "\\", "down"], [40, "\\", "up"]], "key_bindings")
                self.sent, self.window_state = True, "may be open"
                for wait in (SETTLE_S, AGAIN_S, LATE_S):
                    time.sleep(wait)
                    reading = self._read()
                    if reading is not None and reading.open:
                        break
                if reading is None or not reading.open:
                    if reading is not None and not self._changed(behind, self._last):
                        self.window_state = "not opened"
                        self.log(f"  the key bindings: the window didn't open after the backslash (nothing changed "
                                 f"where it opens; {reading.text()}): not read, no Esc")
                    else:
                        self.log("  the key bindings: NOT SEEN after the backslash, "
                                 + ("no frame to tell by" if reading is None else "though the frame changed where it "
                                    "opens") + ": it may be open")
                    return reading
            if reading.read and reading.unsure and not reading.attack:  # (a second frame, then not ATTACK)
                second = self._read()
                if second is not None and second.read:
                    reading = second
            self.log("  the key bindings" + (" (open already)" if already else "") + f": {reading.text()}")
            self._keep(self._frame_read)
            if self._close():
                self.window_state = "closed"
            return reading
        except BaseException:
            if self.window_state == "may be open":  # (the Lie Detector halfway, ...: said before it goes on)
                text = "the KEY BINDINGS window may still be open in the game: close it with Esc or CANCEL"
                self.log(f"KEY CHECK: {text}")
                alarm.page("MapleStory: the key bindings window", text)
            raise

    def _frame(self):
        """The game window's whole client area now (RGB, as the screen shows it after now: Eyes._capture), or None."""
        try:
            frame = self.h.senses.eyes._capture()
        except (BadLook, RuntimeError) as e:
            self.log(f"  the key bindings: no frame ({getattr(e, 'why', e)})")
            return None
        self._last = frame
        return frame

    def _read(self) -> keybindings.Reading | None:
        """A frame now, read (None: no frame)."""
        frame = self._frame()
        if frame is None:
            return None
        reading = self.reader.read(frame)
        if reading.open:
            self._frame_read = frame
        return reading

    @staticmethod
    def _changed(before, after) -> bool:
        """Whether `after` differs from `before` where the window opens (OPEN_AT): CHANGED_SHARE of its px by more
        than CHANGE_LEVEL in a channel. Not told (a frame missing, another size): changed."""
        if before is None or after is None or before.shape != after.shape:
            return True
        (x, y), (w, h) = keybindings.OPEN_AT, keybindings.WINDOW
        a = before[y:y + h, x:x + w, :3].astype(np.int16)
        b = after[y:y + h, x:x + w, :3].astype(np.int16)
        return a.size > 0 and float((np.abs(a - b).max(axis=2) > CHANGE_LEVEL).mean()) >= CHANGED_SHARE

    def _close(self) -> bool:
        """Esc (the window seen open), frames CLOSE_S and CLOSE_AGAIN_S later; Esc once more only if both still show it
        open, and so once more. Whether it's seen closed (said)."""
        act = self.h.actions
        for go in (1, 2):
            act.play([[0, "esc", "down"], [40, "esc", "up"]], "key_bindings_close")
            for wait in (CLOSE_S, CLOSE_AGAIN_S):
                time.sleep(wait)
                after = self._read()
                if after is None:
                    self.log(f"  the key bindings: no frame after Esc x{go}: not known whether it closed")
                    return False
                if not after.open:
                    self.log(f"  the key bindings: closed (Esc x{go}" + (", the first didn't close it" if go == 2
                                                                          else "") + "), nothing saved")
                    return True
        self.log("  the key bindings: STILL OPEN after Esc twice")
        return False

    def _keep(self, frame) -> None:
        """The window's frame saved (lab/shots/keybindings_<time>.png), CHECKS_MAX a run at the most."""
        if frame is None or self.frames_kept >= CHECKS_MAX:
            return
        try:
            from PIL import Image
            paths.SHOTS.mkdir(parents=True, exist_ok=True)
            path = paths.SHOTS / f"keybindings_{time.strftime('%Y%m%d-%H%M%S')}.png"
            Image.fromarray(frame).save(path)
            self.frames_kept += 1
            self.kept = path
        except Exception:
            pass

    # --- 5. what then ----------------------------------------------------------------------------------------------

    def _act(self, doubt: Doubt, reading: keybindings.Reading | None, stars: str = "") -> Outcome:
        """What the window read says, done (see the module's doc, 5.): her keys followed (control.keymap.follow) and
        remembered, then her regular attack's outcome; a stars doubt's out of darts stop raised with her Lucky Seven
        where she has it; her moves' doubt "as configured" with none of them moved."""
        h = self.h
        configured = h.actions.keys.attack.name
        cfg = configured.upper()
        fn = {} if reading is None or not reading.read else self._functions(reading)
        if reading is None or not reading.read or not fn:
            why = "no frame" if reading is None else reading.why if not reading.read else "its keys not read"
            self._unchecked(doubt, f"the key bindings not read ({why})")
            sent = (" (the backslash went out and nothing changed where the window opens: if it's open in the game, "
                    "close it with Esc or CANCEL)" if self.sent else "")
            if doubt.kind == "stars":
                self.log(f"KEY CHECK: the key bindings not read ({why}): out of darts' stop{sent}")
                raise _now(doubt.stop)
            if doubt.kind == "moves":
                text = f"her moves doubted and the key bindings not read ({why}){sent}"
                self.log(f"KEY CHECK: {text.upper()}")
                raise KeyCheckStop(text, at=time.monotonic())
            if doubt.kind in ("casts", "stall", "pet"):
                text = f"HER KEYS: THE KEY BINDINGS NOT READ ({why}): nothing changed{sent}"
                self.log(text)
                return Outcome("held", text, None, reading)
            return self._off(f"ATTACK KEY: THE KEY BINDINGS NOT READ ({why}): no more regular attacks this run "
                             f"(Lucky Seven only){sent}", f"the key bindings not read ({why})", reading)
        preset = f"Key Bindings, preset {reading.preset}" if reading.preset else "Key Bindings"
        f = keymap.follow(h.actions.keys, fn, reading.illegible)
        if f.stop is not None:  # (a move of hers, her Lucky Seven or pet food on no key she can press: not remembered)
            text = f"{f.stop}: her keys can't be followed there"
            self.log(f"HER KEYS: {text.upper()}")
            raise KeyCheckStop(text, at=time.monotonic())
        attack_off = None
        if "attack" in f.off or "attack" in f.unknown:
            attack_off = ("in the tray (on no key)" if reading.in_tray else "on no key" if not reading.attack
                          else f"on {', '.join(k.upper() for k in reading.attack)} (not a key to press)")
        saved = keymap.remember(f.keys, config_keys=getattr(h, "config_keys", None), frame=self.kept,
                                preset=reading.preset, functions=fn, why=doubt.kind,
                                attack_off=None if attack_off is None else f"ATTACK {attack_off}")
        where = (f"remembered in {saved.name} for her next runs" if saved is not None
                 else "not remembered: its file not written")
        others = {k: v for k, v in f.moved.items() if k != "attack"}
        now_on = self._what_on(fn, configured)
        if keymap.changes(h.actions.keys, f.keys):  # (moved, or taken off: her pick-up, chair or sit on no key now)
            h.follow(f.keys)  # (her actions', attack choice's and trips' keys)
            keymap.publish(f.keys, why=f"followed off the window ({doubt.kind} doubt)")
        h.keys_read_at = time.monotonic()  # (her keys as this run's read has them: Hunter.keys_doubted, her pet's care)
        lucky = "lucky_seven" in f.moved
        if lucky and getattr(h, "darts", None) is not None:  # (her presses went on a key that didn't cast: a stretch
            h.darts.followed(time.monotonic())               # of out of darts' from now, whatever else moved)
        # Out of darts' stop due, the window read first: raised once it shows her Lucky Seven where she has it, whatever
        # else it had moved (the review of 2026-10-28: with ATTACK moved or off too, the stop was dropped, DartWatch's
        # `fired` kept, so it never came due again that run)
        stars_stop = doubt.kind == "stars" and not lucky
        followed = keymap.changes_text([(k, a, b) for k, (a, b) in others.items()])
        said = [x for x in f.said if not x.startswith("attack ")
                and not any(x.startswith(k.replace("_", " ") + " on ") for k in f.moved)]
        tail = (f"; HER KEYS FOLLOWED: {followed}" if followed else "") + "".join(f"; {x}" for x in said)
        if "attack" in f.moved:
            key = f.moved["attack"][1]
            text = (f"ATTACK KEY: THE GAME HAS ATTACK ON {key.upper()}, NOT {cfg} ({preset}): her regular attack is "
                    f"{key.upper()} from now on (config.toml unchanged; {where}); {cfg} {now_on}" + tail)
            self.log(text)
            alarm.page("MapleStory: the attack key",
                       f"gamelab found the game's Attack on {key.upper()}, not {cfg}: the hunt uses {key.upper()} for "
                       f"the regular attack ({where})."
                       + (f" Her keys followed too: {followed}." if followed else "")
                       + "".join(f" {x}." for x in said)
                       + f" Put Attack back on {cfg} in the Key Bindings (\\), then `gamelab keys --forget`.")
            if stars_stop:
                self._stars_stop(doubt, preset, followed)
            return Outcome("switched", text, key, reading)
        if attack_off is not None:
            out = self._off(f"ATTACK KEY: THE GAME HAS ATTACK {attack_off.upper()} ({preset}): no more regular attacks "
                            f"this run (Lucky Seven only){tail}", f"ATTACK {attack_off}", reading)
            if stars_stop:
                self._stars_stop(doubt, preset, followed)
            return out
        if others:
            if stars_stop:
                self._stars_stop(doubt, preset, followed)
            text = (f"HER KEYS: THE GAME HAS THEM ON OTHER KEYS ({preset}): followed from now on ({where}): "
                    f"{followed}" + "".join(f"; {x}" for x in said) + f"; attack on {cfg} as configured")
            self.log(text)
            alarm.page("MapleStory: her keys", f"gamelab found her keys moved in the game's Key Bindings and follows "
                                               f"them: {followed} ({where})." + "".join(f" {x}." for x in said)
                       + " Put them back in the Key Bindings (\\), then `gamelab keys --forget`.")
            return Outcome("followed", text, None, reading)
        if stars_stop:
            self._stars_stop(doubt, preset)
        if doubt.kind == "pet":
            food = h.actions.keys.pet_food
            text = (f"HER KEYS: PET FOOD ON {food.name.upper() if food is not None else 'NO KEY'} AS CONFIGURED "
                    f"({preset}): her Pet Food taps took no effect with its key right: out of Pet Food?"
                    + "".join(f"; {x}" for x in said))
            self.log(text)
            return Outcome("as configured", text, configured, reading)
        if doubt.kind in ("moves", "casts", "stall"):
            what = {"moves": "HER MOVES' KEYS", "casts": "HER LUCKY SEVEN AND MOVES' KEYS"}.get(doubt.kind, "HER KEYS")
            text = (f"HER KEYS: {what} AS CONFIGURED ({preset}): the misses were something else; nothing "
                    + ("moved" if said else "changed") + "".join(f"; {x}" for x in said))
            self.log(text)
            return Outcome("as configured", text, configured, reading)
        if said:
            self.log("HER KEYS: " + "; ".join(said))
        if stars == "maybe out":
            return self._off(f"ATTACK KEY: ATTACK IS ON {cfg} AS CONFIGURED ({preset}), AND HER STARS MAY BE OUT: "
                             "no more regular attacks this run (Lucky Seven only: out of darts' own watch stops the "
                             "hunt if her casts keep taking no MP)", "her stars may be out", reading)
        targets = doubt.targets
        h.hold_at(targets)
        text = (f"ATTACK KEY: ATTACK IS ON {cfg} AS CONFIGURED ({preset}): the misses were something else (misses? "
                f"out of reach?); no more regular attacks at that {', '.join(m.kind for m in targets)}")
        self.log(text)
        return Outcome("as configured", text, configured, reading)

    def _stars_stop(self, doubt: Doubt, preset: str, followed: str = "") -> None:
        """A stars doubt's out of darts stop, raised now: the window shows her Lucky Seven where she has it (whatever
        else it had moved, followed already)."""
        lucky = self.h.actions.keys.lucky_seven
        self.log(f"HER KEYS: LUCKY SEVEN ON {lucky.name.upper() if lucky is not None else 'NO KEY'} AS CONFIGURED "
                 f"({preset}" + (f"; followed: {followed}" if followed else "") + "): out of darts' stop")
        raise _now(doubt.stop)

    def _functions(self, reading: keybindings.Reading) -> dict[str, str | None]:
        """What each key does as the window read shows it (perception.keybindings.Reader.functions), {} if not told."""
        try:
            return {} if self._frame_read is None else self.reader.functions(self._frame_read, reading)
        except Exception:
            return {}

    @staticmethod
    def _what_on(fn: dict, key: str) -> str:
        """What the window read shows on `key` now, in words ("does nothing now", "is JUMP now")."""
        what = fn.get(keybindings.SAME_KEY.get(key, key), "?") if fn else "?"
        return "not read" if what == "?" else "does nothing now" if what is None else f"is {what} now"

    def _off(self, text: str, why: str, reading: keybindings.Reading | None = None) -> Outcome:
        """The regular attack off for the run (Hunter.regular_off), said and paged (her next run tries it again:
        control.keymap.at_start)."""
        self.h.regular_off(why)
        self.log(text)
        alarm.page("MapleStory: the attack key", f"gamelab: {text.split(': ', 1)[1]}. Check the game's Key Bindings "
                                                 "(\\) and config.toml. Her next run tries the regular attack again "
                                                 "(5 presses without effect turn it off again).")
        return Outcome("off", text, None, reading)

    def _hold(self, doubt: Doubt, why: str) -> Outcome:
        """No check (`why`): a stars doubt's out of darts stop raised as it was; her moves' doubt logged, nothing done;
        a regular doubt's past CHECKS_MAX the regular attack off for the run (paged), within CHECK_GAP_S held at the
        doubt's targets only (logged)."""
        watch = self.h.attacks.watch
        self._unchecked(doubt, f"no check ({why})")
        if doubt.kind == "stars":
            self.log(f"KEY CHECK: no check ({why}): out of darts' stop")
            raise _now(doubt.stop)
        if doubt.kind in ("moves", "casts", "stall", "pet"):
            text = f"HER KEYS: no check ({why}): {doubt.text().split(' (')[0].lower()}; nothing done"
            self.log(text)
            return Outcome("held", text)
        if len(watch.checks) >= CHECKS_MAX:
            return self._off(f"ATTACK KEY: ANOTHER DOUBT, {why}: no more regular attacks this run (Lucky Seven only)",
                             why)
        self.h.hold_at(doubt.targets)
        text = (f"ATTACK KEY: no check ({why}): no more regular attacks at that "
                f"{', '.join(m.kind for m in doubt.targets)}")
        self.log(text)
        return Outcome("held", text)
