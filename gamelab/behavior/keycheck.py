"""Her regular attack's key doubted, checked and set right: the user's rule of 2026-10-28, after a hunt pressed a
regular attack the game didn't have on its key any more.

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
   dying) and the target in her regular attack's reach (`regular_reach`: the star's, REACH_IN inside reach.STAR_RANGE;
   a wand's box) at its key's look and at the judging one. It took **no effect** when that monster is seen again alive
   with a believable bar as high as before (BAR_STEP/2 under it at the most), no kill of hers placed on it and no EXP
   gain naming its kind since SHOWS_S after the key; an **effect** when its bar is lower, a kill of hers is placed on
   it or a gain names its kind (a party share names none: it isn't one). An effect a Lucky Seven cast at it keyed after
   the press could have made (keyed SHOWS_S before the effect or earlier) is the cast's: the press is passed over, as
   every press not judged is (neither counted nor ending a run). DOUBT_PRESSES no-effects in a row are the doubt.
2. **Her stars** (the thief's: her profile's ammo, "throwing stars"; a magician's regular attack is her wand, nothing
   to run out of): a Lucky Seven cast that took MP since the doubt's first press (behavior.darts.DartWatch.mp_shown_at)
   says she has them; else, with the MP for a cast and a step, the Lucky Seven key and a monster in reach with none on
   that side within the claw's reach and CLAW_SLACK (up close the key swipes, no MP whatever her stars), up to
   STAR_CASTS casts, each judged by her MP bar MP_SHOWS_S after its key: one that takes MP says she has them, STAR_CASTS
   that take none (her bar over darts.LOW_STEPS; not one after which a look saw her touched within KNOCK_S of its key:
   a monster on her the key swiped) are out of darts, behavior.darts.OutOfDarts (the stop as it is: her retreat, the
   page, the recording kept). Without the MP it isn't known; with no monster to cast at it's tried again from the place
   she checks at (3.), then not known; and the check goes on.
3. **A place to do it**: no monster within PLACE_CLEAR of her, she stays; else away to a safe place as on the Lie
   Detector (behavior.retreat.away: her moves alone, 10 s at most).
4. **The window** (`KeyCheck._window`; perception.keybindings reads it): the mouse moved off where it opens (MOUSE_AT:
   a pointer over a cell hides it, and over a preset page draws that page as "checked"), a frame (open already: read,
   no backslash), the backslash tapped, a frame SETTLE_S later, one more AGAIN_S on if it isn't open yet (still not:
   not read, and no Esc: with nothing open Esc opens the game's MENU, the key bindings have ESC on MENU); its frame
   saved (lab/shots/keybindings_<time>.png); Esc only with it seen open, a frame CLOSE_S later, Esc once more if it's
   still seen open, else it's said and paged that it didn't close. Only the backslash, Esc and one mouse move are sent;
   with the Lie Detector alarm up the player refuses them (the alarm goes on as ever; a window left open then is said
   and paged: "close it with Esc or CANCEL").
5. **What then** (`KeyCheck._act`, said in capitals; a telemetry event 'attack_key'):
   - ATTACK on another key the player can press (every key the window binds is one of config.toml's names, and the
     player takes them all): her regular attack is that key for the rest of the run (Hunter.rekey: her Actions' and
     her attack choice's keys; config.toml unchanged), and whoever is at the PC is paged;
   - ATTACK on no key (in the tray, or on none), on a key not to press (Esc, the backslash, maplehelper's hotkeys F1-F4
     and F12: NOT_ATTACK), or the window not read: no more regular attacks this run (Lucky Seven only; her configured
     key may do something else now), paged;
   - ATTACK on her configured key after all: the misses were something else ("misses? out of reach?"): no more
     regular attacks at the doubt's targets (`RegularWatch.no_regular_at`), the hunt goes on, logged only.
   At most CHECKS_MAX checks a run and none within CHECK_GAP_S of the last: a doubt in that time holds the regular
   attack at its targets only; once CHECKS_MAX have run, the next doubt turns it off for the run (paged).

In the hunt loop (`tour --loop`, `--grind`: Hunt.run) the doubt raises KeyDoubt where her health is asked after
(Hunter.health_check), and the hunt loop runs the check (Hunter.check_keys) as it stands still for a look that failed
its checks: its clocks paused over it, the next pass from where she is. Elsewhere (`hunt kill`, `hunt clear`, `nav`, a
plain `tour`, `probe run`) her regular presses are held (Lucky Seven only) and whoever is at the PC is paged to run
`gamelab keys --read`.

On the live data (gamelab README: The regular attack's key checked; tests/test_keycheck.py replays the runs of
2026-10-25 to 28 through `RegularWatch`, tests/fixtures/regular_live_2026-10-28.json): at 00:31 the doubt at the look
of 00:31:19.8, its 5th press keyed at 00:31:18.4, 13 s before her last useless press; with her keys right (the other
15 runs, 245 minutes, 595 regular presses) never, 60 judged, 5 without effect, 2 in a row at the most."""

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from maplehelper.engine import LEFT, RIGHT

from .. import paths, telemetry
from ..attack import reach
from ..control import alarm, client
from ..localization.monsters import TRACK_SLACK, Monster, seen_again
from ..perception import keybindings
from ..perception.framecheck import BadLook

DOUBT_PRESSES = 5      # regular presses in a row that took no effect: the doubt (the user's "5 times")
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
WALK_S = 120.0         # units a second a target unseen for a while may have walked (a Horny Mushroom's 100 or so)
KEPT_S = 8.0           # s her casts, touches, gains and kills are kept for the presses waiting
STAR_CASTS = 5         # Lucky Seven casts that took no MP: out of darts (her stars' test)
MP_SHOWS_S = 0.55      # a test cast judged by her MP bar this long after its key (attacks.MP_SHOWS_S) ...
STAR_TEST_S = 1.5      # ... looked at for this long at the most
STAR_TEST_MAX_S = 8.0  # the test's casts within this long (a cast's way waited for while a monster is up close)
CLAW_SLACK = 50.0      # a test cast's way has no monster within this past the claw's reach (reach.CLAW_FAR): it walks
CHECKS_MAX = 3         # checks a run at the most ...
CHECK_GAP_S = 300.0    # ... none within this of the last
SETTLE_S = 0.6         # s after the backslash before the window's frame (her pet's window's: behavior.pet.SETTLE_S) ...
AGAIN_S = 0.4          # ... and one more this long after, if it isn't open yet
CLOSE_S = 0.6          # s after Esc before the frame that says it closed
MOUSE_AT = (1100, 300)  # client px at 1366x768 the pointer goes to first: the scene right of where it opens
PLACE_CLEAR = (250.0, 120.0)  # no monster this near (x, y) in a look: she checks where she is (hunter.PET_CLEAR)
# Keys the regular attack isn't switched to: Esc (maplehelper's abort, and the game's MENU), the backslash (SET KEY: what
# opens the window), maplehelper's hotkeys (config.toml [hotkeys] and [control]: hidden from the game while it runs)
NOT_ATTACK = ("esc", "\\", "f1", "f2", "f3", "f4", "f12")
KEPT_PRESSES = 12      # a doubt's presses kept for its report (the run's last)
# Her keys besides the regular attack's, for a clash with the key the game has ATTACK on (said and paged)
OTHER_KEYS = ("jump", "lucky_seven", "pick_up", "pet_food", "up", "down", "left", "right", "chair", "sit")


def _believable(bar: float | None) -> bool:
    return bar is not None and 0.0 < bar < 1.0


def regular_reach(s, m: Monster) -> bool:
    """Whether her regular attack reaches monster `m` from where look `s` has her: her profile's (character.active):
    a wand's box (Geometry "melee": its near to far ahead, her body's half aside, its down to up), else the star's
    (reach.can_hit, REACH_IN inside reach.STAR_RANGE: walls and floors stop it)."""
    if s is None or s.where is None or s.platform is None:
        return False
    from .. import character
    geo = character.active().attacks["regular"].geometry
    if geo.kind == "melee" and geo.box is not None:
        b, (_, half) = geo.box, reach.body_of(m.kind)
        ahead, dy = abs(m.x - s.where.x), s.where.y - m.y
        return b.near - half <= ahead <= b.far + half and -b.down <= dy <= b.up
    return reach.can_hit(s.platform, s.where.x, s.where.y, m, reach=reach.STAR_RANGE - REACH_IN)


@dataclass
class Press:
    """A regular press being judged: its key's time, its target (followed look to look) and that one's bar before it,
    and what came of it (`verdict`: "effect", "none" or "passed", and why; `at`: the look it was judged at, or its
    effect's time; `after`: the target's bar then)."""
    t: float
    m: Monster
    bar: float
    seen: float = 0.0         # when `m` was last seen
    verdict: str | None = None
    why: str = ""
    at: float | None = None
    after: float | None = None

    @property
    def kind(self) -> str:
        return self.m.kind


def _clock(t: float) -> str:
    """A time.monotonic time as the wall clock had it (HH:MM:SS.s)."""
    wall = time.time() - (time.monotonic() - t)
    return time.strftime("%H:%M:%S", time.localtime(wall)) + f".{int((wall % 1) * 10)}"


@dataclass
class Doubt:
    """DOUBT_PRESSES regular presses in a row that took no effect (RegularWatch.due): the presses (the run's last
    KEPT_PRESSES), when the last was judged (`known`), the run's how-many-th doubt."""
    presses: list[Press]
    known: float
    number: int = 1

    @property
    def first(self) -> float:
        return self.presses[0].t

    @property
    def targets(self) -> list[Monster]:
        """The monsters its presses went at, each once (as last seen)."""
        out: list[Monster] = []
        for p in self.presses:
            if not any(seen_again(m, [p.m]) is p.m for m in out):
                out.append(p.m)
        return out

    def text(self) -> str:
        kinds = ", ".join(sorted({p.kind for p in self.presses}))
        bars = "/".join(f"{b}%" for b in sorted({round((p.after if p.after is not None else p.bar) * 100)
                                                  for p in self.presses}))
        judged = sorted(p.at - p.t for p in self.presses if p.at is not None)
        each = f", each judged {judged[len(judged) // 2]:.1f} s after its key" if judged else ""
        return (f"THE REGULAR ATTACK TOOK NO EFFECT {len(self.presses)} TIMES IN A ROW ({kinds}, its bar {bars} "
                f"throughout, {_clock(self.presses[0].t)}-{_clock(self.presses[-1].t)}: in reach, untouched{each})")


class KeyDoubt(Exception):
    """Raised by Hunter.health_check in the hunt loop on a doubt (RegularWatch.due): the hunt loop runs the check
    (Hunter.check_keys) and goes on. Not a HuntStop: the hunt doesn't end."""

    def __init__(self, doubt: Doubt):
        self.doubt = doubt
        super().__init__(doubt.text())


@dataclass
class RegularWatch:
    """Her regular presses judged look by look for the doubt (see the module's doc): `pressed`, `cast`, `touched`,
    `heard`, `killed` and `look` feed it; `due` is the doubt, once a run is DOUBT_PRESSES long; `holds` says whether a
    regular press at a target is held. `reach_ok`: whether her regular attack reaches a monster from where a look has
    her (look, monster -> bool: `regular_reach`; None: always)."""
    presses: int = DOUBT_PRESSES
    reach_ok: Callable | None = None
    pending: list[Press] = field(default_factory=list)
    run: list[Press] = field(default_factory=list)       # the no-effects in a row so far
    casts: list[tuple[float, Monster]] = field(default_factory=list)  # Lucky Seven keys at a target, the last KEPT_S
    touches: list[float] = field(default_factory=list)   # the looks that saw her touched, the last KEPT_S
    gains: list[tuple[float, frozenset | None]] = field(default_factory=list)  # her EXP's gains: (frame, kinds)
    kills: list[tuple] = field(default_factory=list)     # her kills placed: (when, kind, platform, x, track)
    counts: Counter = field(default_factory=Counter)     # presses by what came of them (and why passed over)
    longest: int = 0                                     # the longest run of no-effects
    held: bool = False                                   # regular presses held (a doubt waiting for its check)
    off: str | None = None                               # why the regular attack is off for the run (None: it isn't)
    no_regular_at: list[Monster] = field(default_factory=list)  # targets it skips (a check found the key right)
    doubts: list[Doubt] = field(default_factory=list)
    checks: list[float] = field(default_factory=list)    # when each check ran
    _due: Doubt | None = None

    # --- fed ------------------------------------------------------------------------------------------------------

    def pressed(self, t: float, m: Monster | None, *, known: bool = True, dying: bool = False, ground: bool = True,
                roped: bool = False, reach_ok: bool = True) -> None:
        """A regular press keyed at `t` at target `m` (None: nothing seen in its way), with what decides whether it's
        judged: its kind's HP `known`, it not `dying`, from the `ground`, her not `roped` (on a rope or at its top),
        her reach of it at the key's look (`reach_ok`)."""
        why = ("nothing in its way" if m is None else "its HP not known" if not known else "dying" if dying else
               "a jump's" if not ground and not JUMPS_JUDGED else "on a rope" if roped else
               "no bar to judge by" if not _believable(m.bar) else "out of reach" if not reach_ok else "")
        if why:
            self.counts[f"passed: {why}"] += 1
            return
        self.pending.append(Press(t, m, float(m.bar), t))

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
        for p in self.pending:
            if p.verdict is None:
                self._judge(p, t, monsters, s)
        while self.pending and self.pending[0].verdict is not None:
            p = self.pending.pop(0)
            self.counts[p.verdict if p.verdict != "passed" else f"passed: {p.why}"] += 1
            if p.verdict == "effect":
                self.run = []
            elif p.verdict == "none":
                self.run.append(p)
                self.longest = max(self.longest, len(self.run))
                if len(self.run) >= self.presses and self._due is None and not self.held and self.off is None:
                    self._due = Doubt(list(self.run[-KEPT_PRESSES:]), t, len(self.doubts) + 1)
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
        if t - p.t < JUDGE_FROM_S:
            return
        if t - p.t > JUDGE_BY_S:
            p.verdict, p.why = "passed", "not judged in time"
            return
        if again is None or not _believable(again.bar) or again.bar > p.bar + BAR_STEP:
            return  # (not seen, no bar to go by, or one higher than its own was: another monster's)
        if self.reach_ok is not None and s is not None and not self.reach_ok(s, again):
            p.verdict, p.why = "passed", "out of reach at its look"
            return
        p.verdict, p.at, p.after = "none", t, float(again.bar)

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

    def _effect(self, p: Press, t: float, again: Monster | None) -> float | None:
        """When an effect of press `p` (or of a cast after it) shows by `t`: its target's bar lower, a kill of hers
        placed on it, a gain naming its kind; None."""
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

    def holds(self, m: Monster | None) -> bool:
        """Whether a regular press at `m` is held: a doubt waiting for its check, the regular attack off for the run,
        or `m` one of the targets a check found the key right at."""
        if self.held or self.off is not None:
            return True
        return m is not None and any(seen_again(h, [m]) is m for h in self.no_regular_at)

    def checked(self, t: float) -> None:
        """A check ran at `t`: the presses waiting and the run start again."""
        self.checks.append(t)
        self.run, self.pending, self.held = [], [], False

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
        out = (f"her regular presses: {c['effect'] + c['none']} judged, {c['none']} took no effect, {self.longest} in a "
               f"row at the most (the doubt at {self.presses})")
        if self.doubts:
            out += f"; {len(self.doubts)} doubt(s), {len(self.checks)} check(s)"
        if self.off:
            out += f"; the regular attack off: {self.off}"
        return out


# --- the check ----------------------------------------------------------------------------------------------------

@dataclass
class Outcome:
    """What a check found and did: `what` "switched", "off", "as configured" or "held" (no check: CHECKS_MAX, the gap),
    the key the game has ATTACK on (`key`), the window's reading, what was said in capitals, how her stars were told
    (`stars`: "shown", a cast's MP since the doubt's first press; "test", a test cast took MP; "not known", "no target"
    (no monster to cast at, where she was and where she checked); "no stars", a magician), and how long it took."""
    what: str
    text: str
    key: str | None = None
    reading: keybindings.Reading | None = None
    stars: str = ""
    seconds: float = 0.0


class KeyCheck:
    """The check after a doubt (see the module's doc) by hunter `h`: its senses, actions, attack choice, out of darts'
    watch and its log. `reader`: the window's (perception.keybindings.Reader; made when first needed). The simulator's
    eyes draw the window from the world's bindings (sim.fakes.FakeEyes._capture)."""

    def __init__(self, h, reader: keybindings.Reader | None = None, log: Callable[[str], None] | None = None):
        self.h = h
        self.log = log or h.log
        self._reader = reader
        self.frames_kept = 0
        self._frame_read = None  # the frame the window was last read on

    @property
    def reader(self) -> keybindings.Reader:
        if self._reader is None:
            self._reader = keybindings.Reader()
        return self._reader

    def run(self, doubt: Doubt) -> Outcome:
        """The check: her stars, a place, the window, what then (each said). Raises OutOfDarts (her stars' test) and
        client.LieDetectorAlarm (the player refuses its keys then)."""
        start = time.monotonic()
        watch = self.h.attacks.watch
        if (why := watch.may_check(start)) is not None:
            return self._hold(doubt, why)
        watch.checked(start)
        self.h.hold_regular(False)  # (the doubt's hold let go: what the check finds decides)
        stars = self._stars(doubt)
        self._place()
        if stars == "no target":  # (from the place she checks at, once more)
            if (stars := self._stars(doubt)) == "no target":
                self.log("  her stars: not known (no monster to cast at from here either)")
        reading = self._window()
        out = self._act(doubt, reading)
        out.stars, out.seconds = stars, time.monotonic() - start
        telemetry.emit("event", kind="attack_key", text=out.text, what=out.what, key=out.key, stars=stars,
                       seconds=round(out.seconds, 2), **({"reading": reading.record()} if reading is not None else {}))
        return out

    # --- 1. her stars ----------------------------------------------------------------------------------------------

    def _stars(self, doubt: Doubt) -> str:
        """Whether she has her stars (the thief's: see the module's doc); raises OutOfDarts when STAR_CASTS test casts
        took no MP. Returns what told it."""
        from .. import character
        from ..attack import choice
        from .darts import LOW_STEPS, OutOfDarts
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
        end = time.monotonic() + STAR_TEST_MAX_S
        while casts < STAR_CASTS and time.monotonic() < end:
            s = self.h.look()
            mp = a.mp
            if mp is None or not choice.can_cast(mp, a.max_mp, a.skill.mp + a.max_mp / choice.MP_STEPS):
                have = "not read" if mp is None else f"{max(0.0, choice.mp_points(mp, a.max_mp)):.0f}"
                self.log(f"  her stars: not known (MP {have} < a cast's {a.skill.mp:.0f} and a step)")
                return "not known"
            side = self._cast_side(s, doubt)
            if side is None:
                if not casts:
                    self.log("  her stars: not known yet (no monster in reach to cast at, out of the claw's)")
                    return "no target"
                time.sleep(0.2)  # (the monsters walk: another look)
                continue
            if not self._cast(side):
                time.sleep(0.2)
                continue
            casts += 1
            key = self.h.actions.attack_at
            if self._mp_shown(key):
                self.log(f"  her stars: the test's cast {casts} took MP: she has them")
                return "test"
            if (hit := self.h.hit_at) is not None and hit >= key - KNOCK_S:  # (one on her: the key may have swiped it)
                continue
            took += 1
            if darts.share * choice.MP_STEPS < LOW_STEPS:
                self.log("  her stars: not known (her MP too low to tell)")
                return "not known"
        if took < STAR_CASTS:
            self.log(f"  her stars: not known ({casts} test casts in {STAR_TEST_MAX_S:.0f} s, {took} of them took no MP "
                     f"untouched; {STAR_CASTS} needed)")
            return "not known"
        stop = OutOfDarts(took, time.monotonic() - doubt.first, darts.share, at=time.monotonic())
        stop.args = (f"the regular attack took no effect {len(doubt.presses)} times in a row; then {took} Lucky Seven "
                     f"casts took no MP (her MP {darts.share:.0%}): she's out of throwing stars",)
        raise stop

    def _cast_side(self, s, doubt: Doubt) -> str | None:
        """Which way a test cast goes: at a monster in reach from where she is in look `s`, out of the claw's (up close
        the key swipes, no MP), she on a platform, not on a rope; the doubt's last target's way first. None if
        there's none."""
        a = self.h.attacks
        if s is None or s.where is None or s.platform is None or self.h.on_rope(s):
            return None
        x, y = s.where.x, s.where.y
        first = RIGHT if doubt.presses[-1].m.x > x else LEFT
        for side in (first, LEFT if first == RIGHT else RIGHT):
            if any((m.x > x) == (side == RIGHT) and abs(m.x - x) < reach.CLAW_FAR + CLAW_SLACK and abs(m.y - y) < 60
                   for m in s.monsters):  # (one that near: the key may swipe it, no MP whatever her stars)
                continue
            m = a.target(side)
            if m is not None and reach.can_hit(s.platform, x, y, m):
                return side
        return None

    def _cast(self, side: str) -> bool:
        """One Lucky Seven press that way, forced (Attacks.forced: chosen, logged and its MP reckoned as any cast)."""
        a, act = self.h.attacks, self.h.actions
        saved = act.skill, act.choose
        act.skill, act.choose, a.forced = True, a.key, "lucky_seven"
        try:
            return act.throw(side) is not False
        finally:
            act.skill, act.choose, a.forced = saved[0], saved[1], None

    def _mp_shown(self, key: float) -> bool:
        """Looks after a cast keyed at `key`, STAR_TEST_S at the most: whether her bar showed its MP by MP_SHOWS_S
        after it (DartWatch.mp_shown_at after the key)."""
        darts = self.h.darts
        while time.monotonic() < key + STAR_TEST_S:
            self.h.look()
            if darts.mp_shown_at is not None and darts.mp_shown_at > key:
                return True
            if darts.read_at is not None and darts.read_at >= key + MP_SHOWS_S:
                return False
        return False

    # --- 2. a place ------------------------------------------------------------------------------------------------

    def _place(self) -> None:
        """No monster within PLACE_CLEAR of her in a look: she stays; else away to a safe place (behavior.retreat)."""
        from .retreat import KEY_CHECK, away
        s = self.h.look()
        near = [] if s.where is None else [m for m in s.monsters if abs(m.x - s.where.x) < PLACE_CLEAR[0]
                                           and abs(m.y - s.where.y) < PLACE_CLEAR[1]]
        if not near:
            self.log(f"  the key check where she is: no monster within {PLACE_CLEAR[0]:.0f} of her")
            return
        self.log(f"  the key check: {', '.join(m.kind for m in near)} near her: to a safe place first")
        away(self.h, time.monotonic(), KEY_CHECK, self.log)

    # --- 3. the window ---------------------------------------------------------------------------------------------

    def _window(self) -> keybindings.Reading | None:
        """The KEY BINDINGS window opened, read and closed (see the module's doc): its reading (None: no frame)."""
        client.check_alarm()
        act = self.h.actions
        seen_open = False
        try:
            frame = self._frame()
            if frame is None:
                self.log("  the key bindings: no frame of the game to read them on: not opened")
                return None
            mouse = (round(MOUSE_AT[0] * frame.shape[1] / keybindings.FRAME[0]),
                     round(MOUSE_AT[1] * frame.shape[0] / keybindings.FRAME[1]))
            act.play([[0, "mouse", *mouse]], "key_bindings_mouse")
            reading = self._read()
            already = reading is not None and reading.open
            if not already:
                act.play([[0, "\\", "down"], [40, "\\", "up"]], "key_bindings")
                time.sleep(SETTLE_S)
                reading = self._read()
                if reading is None or not reading.open:
                    time.sleep(AGAIN_S)
                    reading = self._read()
            seen_open = reading is not None and reading.open
            if not seen_open:
                self.log("  the key bindings: the window didn't open after the backslash"
                         + ("" if reading is None else f" ({reading.text()})") + ": not read, no Esc")
                return reading
            if reading.read and reading.unsure and not reading.attack:  # (a second frame, then not ATTACK)
                second = self._read()
                if second is not None and second.read:
                    reading = second
            self.log("  the key bindings" + (" (open already)" if already else "") + f": {reading.text()}")
            self._keep(self._frame_read)
            seen_open = not self._close()
            return reading
        finally:
            if seen_open:
                text = "the KEY BINDINGS window may still be open in the game: close it with Esc or CANCEL"
                self.log(f"KEY CHECK: {text}")
                alarm.page("MapleStory: the key bindings window", text)

    def _frame(self):
        """The game window's whole client area now (RGB, as the screen shows it after now: Eyes._capture), or None."""
        try:
            return self.h.senses.eyes._capture()
        except (BadLook, RuntimeError) as e:
            self.log(f"  the key bindings: no frame ({getattr(e, 'why', e)})")
            return None

    def _read(self) -> keybindings.Reading | None:
        """A frame now, read (None: no frame)."""
        frame = self._frame()
        if frame is None:
            return None
        reading = self.reader.read(frame)
        if reading.open:
            self._frame_read = frame
        return reading

    def _close(self) -> bool:
        """Esc (the window seen open), a frame CLOSE_S later; Esc once more if it's still seen open. Whether it's seen
        closed (said)."""
        act = self.h.actions
        for go in (1, 2):
            act.play([[0, "esc", "down"], [40, "esc", "up"]], "key_bindings_close")
            time.sleep(CLOSE_S)
            after = self._read()
            if after is None:
                return False
            if not after.open:
                self.log(f"  the key bindings: closed (Esc x{go}), nothing changed")
                return True
        return False

    def _keep(self, frame) -> None:
        """The window's frame saved (lab/shots/keybindings_<time>.png), CHECKS_MAX a run at the most."""
        if frame is None or self.frames_kept >= CHECKS_MAX:
            return
        try:
            from PIL import Image
            paths.SHOTS.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(paths.SHOTS / f"keybindings_{time.strftime('%Y%m%d-%H%M%S')}.png")
            self.frames_kept += 1
        except Exception:
            pass

    # --- 4. what then ----------------------------------------------------------------------------------------------

    def _act(self, doubt: Doubt, reading: keybindings.Reading | None) -> Outcome:
        h = self.h
        configured = h.actions.keys.attack.name
        cfg = configured.upper()
        if reading is None or not reading.read:
            why = "no frame" if reading is None else reading.why
            return self._off(f"ATTACK KEY: THE KEY BINDINGS NOT READ ({why}): no more regular attacks this run "
                             f"(Lucky Seven only)", f"the key bindings not read ({why})", reading)
        preset = f"Key Bindings, preset {reading.preset}" if reading.preset else "Key Bindings"
        if any(keybindings.same_key(k, configured) for k in reading.attack):
            targets = doubt.targets
            h.attacks.watch.no_regular_at.extend(targets)
            text = (f"ATTACK KEY: ATTACK IS ON {cfg} AS CONFIGURED ({preset}): the misses were something else (misses? "
                    f"out of reach?); no more regular attacks at that {', '.join(m.kind for m in targets)}")
            self.log(text)
            return Outcome("as configured", text, configured, reading)
        on = [k for k in reading.attack if k not in NOT_ATTACK]
        if not on:
            where = ("in the tray (on no key)" if reading.in_tray else "on no key" if not reading.attack else
                     f"on {', '.join(k.upper() for k in reading.attack)} (not a key to press)")
            return self._off(f"ATTACK KEY: THE GAME HAS ATTACK {where.upper()} ({preset}): no more regular attacks "
                             f"this run (Lucky Seven only)", f"ATTACK {where}", reading)
        key = on[0]
        now_on = self._what_on(reading, configured)
        clash = [f for f in OTHER_KEYS if (k := getattr(h.actions.keys, f, None)) is not None
                 and keybindings.same_key(k.name, key)]
        h.rekey(key)
        text = (f"ATTACK KEY: THE GAME HAS ATTACK ON {key.upper()}, NOT {cfg} ({preset}): her regular attack is "
                f"{key.upper()} for the rest of this run (config.toml unchanged); {cfg} {now_on}"
                + (f"; {key.upper()} is her {', '.join(clash)} key too" if clash else ""))
        self.log(text)
        alarm.page("MapleStory: the attack key",
                   f"gamelab found the game's Attack on {key.upper()}, not {cfg}: the hunt uses {key.upper()} for the "
                   f"regular attack this run. Put Attack back on {cfg} in the Key Bindings (\\), or set config.toml."
                   + (f" {key.upper()} is her {', '.join(clash)} key too." if clash else ""))
        return Outcome("switched", text, key, reading)

    def _what_on(self, reading: keybindings.Reading, key: str) -> str:
        """What the window read shows on `key` now, in words ("does nothing now", "is JUMP now")."""
        try:
            fn = {} if self._frame_read is None else self.reader.functions(self._frame_read, reading)
        except Exception:
            fn = {}
        what = fn.get(keybindings.SAME_KEY.get(key, key), "?") if fn else "?"
        return "not read" if what == "?" else "does nothing now" if what is None else f"is {what} now"

    def _off(self, text: str, why: str, reading: keybindings.Reading | None = None) -> Outcome:
        """The regular attack off for the run (Hunter.regular_off), said and paged."""
        self.h.regular_off(why)
        self.log(text)
        alarm.page("MapleStory: the attack key", f"gamelab: {text.split(': ', 1)[1]}. Check the game's Key Bindings "
                                                 "(\\) and config.toml.")
        return Outcome("off", text, None, reading)

    def _hold(self, doubt: Doubt, why: str) -> Outcome:
        """No check (`why`): past CHECKS_MAX the regular attack off for the run (paged); within CHECK_GAP_S held at the
        doubt's targets only (logged)."""
        watch = self.h.attacks.watch
        if len(watch.checks) >= CHECKS_MAX:
            return self._off(f"ATTACK KEY: ANOTHER DOUBT, {why}: no more regular attacks this run (Lucky Seven only)",
                             why)
        watch.no_regular_at.extend(doubt.targets)
        self.h.hold_regular(False)  # (held at its targets only)
        text = (f"ATTACK KEY: no check ({why}): no more regular attacks at that "
                f"{', '.join(m.kind for m in doubt.targets)}")
        self.log(text)
        return Outcome("held", text)
