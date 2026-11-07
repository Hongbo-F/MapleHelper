"""The safety stop: her HP, in HP, under a reserve of touches (`Reserve`: her profile's hits times the largest touch
the map's monsters can deal her) for a moment (two looks DEBOUNCE_EXACT_S apart by her HP number, DEBOUNCE_S by her
bar), or under one touch (the floor) on FLOOR_LOOKS looks in a row, ends the hunt, raised at the look or glance it's
due at. She goes to a safe place as she does on the Lie Detector (behavior.retreat: `on_alarm` around every hunting
command), the log says why in capitals with the numbers, and whoever is at the PC is paged as the Lie Detector pages
them (control.alarm.page: alert sounds and a Windows notification that doesn't take focus from the game). No sitting
and no waiting for HP: the user, 2026-10-26, "Just remove the sit recover hp functionality, it's no longer needed". Her
pet drinks her potions.

The rule is the user's, 2026-10-28 ~12:10: "The safety stop should be tuned, it should be HP under the amount of taking
3 hits from the monsters on the map", and at once after: "Well for thief with high evasion this should be 2 hits, 3 for
magician". So the hits are her profile's (gamelab.character's `safety_hits`: the magician 3; the thief 3 for now, the
user's "3 hits for now" of ~15:30 until her retreat dodges, then 2: gamelab.classes.thief.SAFETY_HITS), and the
threshold is absolute HP, per map and per character. At ~15:40: "please make sure the '3 hits' is dynamically
calculated from the map's strongest mob, not calculated once and hard-coded". So nothing here or in the hunter holds a
touch's size: the reserve is worked out live (Hunter._reserve_check, at every look) from
- the map's kinds (its spawn list's monsters, MapModel.mob_damage: `kinds_of`) and any other kind a look sees on it
  (its PADamage from the game data's monster table: `mob_touch`);
- each kind's touch: the game's own damage-taken formula (attack.danger.touch_taken: this game's client as two
  readings of it have it, meowdb's and the osms datamine's) at the kind's PADamage (the game data's), her level and
  her DEF as perception has them (behavior.bootstrap: her HUD's "LV." digits, else her EXP number; her stat window's
  W.DEF, else fitted to her touches, else DEF 0, the most a touch takes), the roll's top (`Touch.prior`);
- her own exact touches by that kind (her HP number's drops: perception.hp_number, kept in `Touches`), those at her
  level and her last CARRY_N at others carried to it by the formula's level term (`Touches.measured`: a level-up keeps
  them), which check it: MARGIN over the largest, if that's more (the game hitting harder than the formula says), from
  the first one; they raise it, never lower it (`Touch.hp`);
and made again, said in the log with why, whenever any of it changes: the map, a kind seen, her level, her max HP, her
DEF, a touch over it. A kind is told by the touch's size (`kind_of_touch`), not by the monster nearest her (none was
seen near her at 853 of 1,220 touches); one whose size fits no kind (a drop over frames not read: the perception
service forgets her number across them, worldstate.HP_GAP_S; a kind the data hasn't) counts for none, nor one taken
before her level was read. Her max HP is her HUD's number's: a command that hunts waits HP_WAIT_S at its start for it
and hunts nothing without it (`HpNotRead`, unless `--low-hp`): no share of her bar is her hits of the map's strongest
touch on every map and at every level. Elsewhere (a `nav` trip, a hunter whose eyes draw no HP number) the threshold is
UNREAD_SHARE of her bar until it reads (behavior.bootstrap), said. A map whose monsters' touches the game data doesn't
give: FALLBACK_SHARE of her max HP, said.

Why the old rule went (65% of her HP for 2 s over 3 looks, or 25% two looks in a row). It rested on her pet drinking an
Orange Potion (250 HP of her 774) whenever a touch left her under its 70%, which put her back over 65% within tenths
of a second. By 2026-10-28 she was level 28, her max HP 840 (the HUD's "HP [840/840]"; config.toml's [character] hp
then still 774, level 25's: why her HP and level are perception's now), and her pet's potion +100 that night (the
user's "100"; the review read +160 on the QA walk's recordings from 11:31) against a Horny Mushroom's touch of 103-122
(38 read exactly; 110-125 by the user): each touch under 70% left her a net 3-31 HP lower, and 65% was a place she
stayed with nothing wrong. At ~01:20: "don't worry about the safety stop, it's because the hit damage (110-125) is more
than the HP portion recover amount (100), the pet is still drinking HP potion on each hit".

The numbers (2026-10-28, offline: D:/MapleHelper-scratch/safety3/design; her HP number read exactly on all 50,193
frames of the 17 recordings of 10-24 23:50 to 10-28 12:16, 299 changes of it, 178 touches; the perception service's
bar about 8 a second, 1,204 drops; the hunter's looks of the 87 Tree Dungeon tour and nav runs since 10-25 with 5 looks
or more, 52,678):
- Her touches fit the formula at one DEF a gear (bootstrap.fit_defence on them: tests/fixtures/
  touches_2026-10-28.jsonl): 61.5-70.25 at level 26, 106.5-115.25 at 28 (40 of her last 40 each). At level 28 the
  reserve's DEF is the fit's low end, where a Horny Mushroom's top (123) is under her largest measured touch (122)
  with MARGIN, so the measured one counts. MARGIN: the largest of 38 (122) against the fitted formula's top (123),
  and a bar px.
- Her HP: the HUD's number when it was read at its last change (Hunter: Scene.hp_of, the perception service's world
  step reading it on every change), else the bar less a px (BAR_SLACK: the bar's red px is ceil(104 cur / max) or one
  more outside its animation, so less a px it's under her HP or over by under a px, 8 HP).
- The debounce: her pet's potion lands 0.04-0.20 s after the touch it answers (61 drinks; one frame's median); the bar
  lags the number by 0.07-0.25 s and settles within a px by 0.37 s; two touches came 1.38 s apart at the least (25 under
  5 s; seven at 1.47-1.50 s). No misread in 240,069 looks and frames since 10-25 (the 63 one-frame dips of the bar were
  touches its potion answered, the number says). So under it on two readings DEBOUNCE_EXACT_S apart by her HP number
  (the potion waited for), DEBOUNCE_S by her bar (the potion and the bar's settling); a reading at or over it starts
  again; under the floor (one touch) FLOOR_LOOKS readings in a row stop it at once. Her looks and her glances are its
  readings (a chain's watch glances every 30 ms or so and looks for seconds at times: Hunter._glanced); the looks' gaps
  were a median 0.13 s, p90 0.94, p99 2.78, at most 10.7. An HP read 0 after a higher one isn't believed until the
  next look reads it too (Hunter._doubt_hp; a glance's isn't taken), as before.
- Replayed on the live looks since 10-25 (D:/MapleHelper-scratch/safety3/fix/replay_fix.py): the old rule stops once,
  at 10-28 01:00:55.85 (64% for 2.07 s, 7 looks: the live stop exactly), the reserve never, at 2 hits or 3 with her
  DEF as her touches fit it then; her lowest look 409 HP (10-26 01:07:54, level 25), at level 28 461 (01:00:50.9).

What it costs when it fires (the review of 2026-10-28, on the branch's 2 hits: D:/MapleHelper-scratch/safety3/fix,
stops3.py; the simulator with her pet out of potions on Tree Dungeon; README, The safety stop in HP): at the stop she
has one to two touches left by the rule's own count, and her way to a safe place, her moves alone keeping clear of
monsters as every trip does (waiting them out, no dodging: the user, 2026-10-28, watching her, "there was no attempt
to even evade"), took 0-4 touches. She died on it in 7-8 of 40 runs as live (17 s respawns, 1.25 s invulnerability)
and 13-18 of 42 in the crowd, where the old rule, stopping at about half her HP, lost 3 and 3. Three hits for the
thief left her two to three touches: 2-5 and 7-9, and the user chose 3 for now (README: this code's own runs).

With a potion key of her own drinking under `--potion-hp` (none bound since 2026-10-23; the first map's simulator binds
one, no pet), UNDER under that if it's lower (`HpWatch.threshold`): between the two is where her key keeps her HP.
`--low-hp` overrides it all: 1 or under, a share of her max HP; over 1, HP. A magician's Magic Guard isn't taken off a
touch (Hunter.guard 0, Reserve's `guard` for when something reads whether it's up: it's a toggle, and nothing does).

It's watched on every look and glance (Hunter._vitals, Hunter._glanced: whoever took it), and raised at the one it's
due at (Hunter._stop_now: STOP_ON_LOOK), once a hunt has asked after her health (Hunter.health_check, which raises it
too), not while she retreats, and once: what follows it (her way to the safe place, the stand-by) goes on. A reserve
at or over her pet's warning at her max HP would stop her at once: a command that hunts refuses the map then
(TooStrong: "too strong for her here"; before it hunts if her max HP is read by then, else at the look that reads it),
unless `--low-hp` is given.

Out of darts (behavior.darts) ends a hunt the same way: both are a HuntStop, which retreat.on_alarm, the command line
and the simulator take as one."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ..attack.danger import TOP_U, touch_range, touch_taken
from ..perception.kinds import kind_of
from .bootstrap import FIT_N, UNREAD_SHARE

HITS = 3              # her profile's hits by default (gamelab.character's safety_hits)
DEBOUNCE_S = 0.6      # s: under the threshold on two looks this far apart at least stops the hunt (her bar's) ...
DEBOUNCE_EXACT_S = 0.25  # ... or this far, her HP number's (the bar's settling not waited for: her pet's potion only)
LOOKS = 2             # ... (this many looks)
FLOOR_TOUCHES = 1     # ... and under this many touches ...
FLOOR_LOOKS = 2       # ... on this many looks in a row, however close, stops it then
MARGIN = 0.05         # over the largest of her measured touches by a kind at her level, once one is
FALLBACK_SHARE = 0.35  # of her max HP: the threshold on a map whose monsters' touches the game data doesn't give
BAR_SLACK = 1 / 104   # the bar's share less this (a px at 1366x768) when her HP number isn't read
UNDER = 0.05          # with a potion key of her own (`--potion-hp`), this under its threshold if that's lower
KIND_DEF = (0.5, 2.0)  # a touch is a kind's if the formula gives it at her DEF's ends times these (kind_of_touch) ...
KIND_DEF_TOP = 300.0  # ... or at any DEF up to this while hers isn't known
EXIT = 4              # the command's exit code after it (the Lie Detector's is 3)
MOB_TABLE = "mobs.json"  # the game data's monster table (gamedata/data/out/tables: name, PADamage, ...), for a kind
                         # seen on a map whose spawn list hasn't it (`mob_touch`)
CARRY_N = FIT_N       # her last this many kept touches count at another level than theirs (Touches.measured)
HP_WAIT_S = 5.0       # s a hunting command waits at its start for her HP number (`HpNotRead`)


@dataclass(frozen=True)
class Touch:
    """A kind's touch as the safety stop counts it: `prior`, the formula's top at her DEF and level (the reserve's);
    `measured`, the largest of her exact touches by it as it would be at her level, of `n` (kept: `Touches.measured`:
    those at her level and her last CARRY_N at others, carried), taken at level `taken_at` (None: hers). `hp`: the
    prior, or MARGIN over the measured if that's more, from the first one (a measured touch only ever raises it, so
    none is waited for). `name`: the kind's as the game data has it ("Horny Mushroom"), `pad` its PADamage; `seen`: a
    kind a look saw that the map's spawn list hasn't."""
    kind: str
    prior: int
    measured: int | None = None
    n: int = 0
    name: str = ""
    pad: float = 0.0
    seen: bool = False
    taken_at: int | None = None

    @property
    def said(self) -> str:
        return self.name or self.kind

    @property
    def hp(self) -> int:
        if self.measured is None:
            return self.prior
        return max(self.prior, math.ceil((1 + MARGIN) * self.measured))

    def source(self, defence: float, level: float) -> str:
        """Where its HP comes from, in words."""
        prior = f"the formula's top at PADamage {self.pad:g}, DEF {defence:g}, level {level:g}"
        if self.measured is None:
            return prior
        if self.taken_at is None or self.taken_at == level:
            mine = f"{self.n} measured at level {level:g}, the largest {self.measured}"
        else:
            mine = (f"{self.n} measured, the largest {self.measured} at level {level:g} (taken at level "
                    f"{self.taken_at}, carried by the formula's level term)")
        if self.hp > self.prior:
            return f"{mine}, {MARGIN:.0%} over it (over {prior}: {self.prior})"
        return f"{prior}; {mine}"


@dataclass(frozen=True)
class Reserve:
    """The safety stop's reserve on one map for her: `hits` touches of the largest one (`touches`, by kind, the
    largest first: none if the game data gives no touch for the map's monsters, FALLBACK_SHARE of her max HP then), at
    her `level` and `defence` (the DEF its touches are worked out at: bootstrap.Defence.low), `stats` saying where her
    stats came from (bootstrap.HerStats.describe); `guard`: her Magic Guard's share of a touch to her MP while it's up
    (0: none)."""
    hits: int = HITS
    touches: tuple[Touch, ...] = ()
    level: float = 0.0
    defence: float = 0.0
    guard: float = 0.0
    map_id: int | None = None
    stats: str = ""

    @property
    def touch(self) -> Touch | None:
        """The largest touch, None without game data."""
        return self.touches[0] if self.touches else None

    def touch_hp(self, mp: float | None = None) -> float | None:
        """The largest touch's HP, less her Magic Guard's share while her MP (`mp`, in MP; None: not known) pays for
        the hits; None without game data."""
        t = self.touch
        if t is None:
            return None
        if self.guard and mp is not None and mp >= self.guard * t.hp * self.hits:
            return t.hp * (1 - self.guard)
        return float(t.hp)

    def hp(self, max_hp: float | None, mp: float | None = None) -> float:
        """The threshold at her max HP `max_hp`, in HP; with her max HP not read (None), UNREAD_SHARE of her bar (a
        share: HpWatch counts in shares then)."""
        if max_hp is None:
            return UNREAD_SHARE
        touch = self.touch_hp(mp)
        return FALLBACK_SHARE * max_hp if touch is None else self.hits * touch

    def floor(self, max_hp: float | None, mp: float | None = None) -> float:
        """The floor: FLOOR_TOUCHES touches, in HP (without game data, or her max HP not read, as many of the
        threshold's hits)."""
        touch = self.touch_hp(mp)
        if max_hp is None or touch is None:
            return FLOOR_TOUCHES * self.hp(max_hp, mp) / self.hits
        return FLOOR_TOUCHES * touch

    def words(self, max_hp: float | None, mp: float | None = None) -> str:
        """The threshold in words: "3 Horny Mushroom touches of 129"."""
        if max_hp is None:
            return f"{UNREAD_SHARE:.0%} of her HP bar (her max HP not read yet)"
        touch = self.touch_hp(mp)
        if touch is None:
            return f"{FALLBACK_SHARE:.0%} of her {max_hp:.0f} HP (no game data for the map's monsters' touches)"
        return (f"{self.hits} {self.touch.said} touch{'es' if self.hits != 1 else ''} of {touch:.0f}"
                + (f" ({self.touch.hp} less Magic Guard's {self.guard:.0%})" if touch < self.touch.hp else ""))

    def floor_words(self, mp: float | None = None) -> str:
        """The floor in words: "one Horny Mushroom touch"."""
        n = "one" if FLOOR_TOUCHES == 1 else str(FLOOR_TOUCHES)
        if self.touch_hp(mp) is None:
            return f"{n} of the threshold's {self.hits} hits"
        return f"{n} {self.touch.said} touch{'es' if FLOOR_TOUCHES != 1 else ''}"

    def limit(self, max_hp: float | None, mp: float | None = None) -> str:
        """The threshold alone: "under 387 HP of 840 (46%: 3 Horny Mushroom touches of 129)"."""
        th = self.hp(max_hp, mp)
        if max_hp is None:
            return f"under {th:.0%} of her HP bar (her max HP not read yet)"
        return f"under {th:.0f} HP of {max_hp:.0f} ({th / max(max_hp, 1):.0%}: {self.words(max_hp, mp)})"

    def describe(self, max_hp: float | None, mp: float | None = None) -> str:
        """The reserve as the log says it: its threshold, where its touch comes from, its floor, her stats' sources."""
        out = self.limit(max_hp, mp)
        if max_hp is not None and (t := self.touch) is not None:
            out += f"; {t.said} {t.hp}: {t.source(self.defence, self.level)}" + (" (seen on the map)" if t.seen else "")
            others = ", ".join(f"{o.said} {o.hp}" for o in self.touches[1:])
            if others:
                out += f"; then {others}"
        floor = self.floor(max_hp, mp)
        out += (f"; at once under {floor:.0%} of her bar two looks in a row" if max_hp is None
                else f"; at once under {floor:.0f} two looks in a row")
        return out + (f"; {self.stats}" if self.stats else "")


def kinds_of(model) -> dict[str, tuple[float, str]]:
    """The map's kinds whose touch the game data gives (its spawn list's: MapModel.mob_damage), by kind: the largest
    touch damage (PADamage) of its names, and that name."""
    out: dict[str, tuple[float, str]] = {}
    for name, pad in getattr(model, "mob_damage", {}).items():
        if pad > 0:
            k = kind_of(name)
            out[k] = max(out.get(k, (0.0, "")), (float(pad), name))
    return out


_MOBS: list[dict] | None = None


def mob_touch(kind: str) -> tuple[float, str] | None:
    """A kind's touch damage by the game data's monster table (MOB_TABLE: the largest PADamage of the monsters of that
    kind's name, a stronger one sharing the name counted), and its name; None if the table hasn't it (or isn't here)."""
    global _MOBS
    if _MOBS is None:
        try:
            from gamedata.paths import OUT
            _MOBS = json.loads((OUT / "tables" / MOB_TABLE).read_text(encoding="utf-8"))
        except Exception:  # (no game data on this computer: nothing known of a kind seen)
            _MOBS = []
    pads = [(float(r.get("PADamage") or 0), r["name"]) for r in _MOBS
            if isinstance(r, dict) and r.get("name") and kind_of(r["name"]) == kind]
    best = max(pads, default=None)
    return best if best is not None and best[0] > 0 else None


def kind_of_touch(hp: float, kinds: dict[str, tuple[float, str]], defence: tuple[float, float | None],
                  level: float) -> str | None:
    """The kind of `kinds` (kind -> (PADamage, name): kinds_of) a touch of `hp` was, by its size: the kinds whose touch
    the formula gives at her level and a DEF between her DEF's ends (`defence`: low, high) times KIND_DEF (up to
    KIND_DEF_TOP while there's no high: hers not known), the largest of them if several; None if none (it counts for
    none then)."""
    low, high = defence
    lo_def, hi_def = low * KIND_DEF[0], (KIND_DEF_TOP if high is None else high * KIND_DEF[1])
    fits = [(pad, k) for k, (pad, _) in kinds.items()
            if touch_taken(pad, hi_def, level, 0.0) <= hp <= touch_taken(pad, lo_def, level, TOP_U)]
    return max(fits)[1] if fits else None


def reserve_of(kinds: dict[str, tuple[float, str]], hits: int = HITS, defence: float = 0.0, level: float = 1.0,
               measured: dict[str, tuple] | None = None, guard: float = 0.0, map_id: int | None = None,
               seen=frozenset(), stats: str = "") -> Reserve:
    """Her reserve over `kinds` (kind -> (PADamage, name): the map's, kinds_of, and those seen on it, mob_touch):
    `hits` of the largest touch, each kind's the formula's top at `defence` and `level` (touch_range), or her
    `measured` touches by it as they'd be at that level (kind -> (the largest, how many[, the level it was taken at]):
    Touches.measured) as Touch says. `seen`: the kinds a look saw that the map's spawn list hasn't; `stats`: where her
    stats came from, in words."""
    measured = measured or {}
    out = []
    for k, (pad, name) in kinds.items():
        most, n, *at = measured.get(k, (None, 0))
        out.append(Touch(k, touch_range(pad, defence, level)[1], most, n, name, pad, k in seen,
                         at[0] if at else None))
    out.sort(key=lambda t: (-t.hp, t.kind))
    return Reserve(hits, tuple(out), level, defence, guard, map_id, stats)


def carried(hp: int, taken_at: float, level: float) -> int:
    """The most a touch of `hp` taken at level `taken_at` can be at `level` by the formula (attack.danger.touch_taken:
    level / 10 off before it's cut to whole HP): HP less the whole levels' tenths gone (none for under 10 levels up; one
    more for a level lower)."""
    return int(hp - math.floor((level - taken_at) / 10))


class Touches:
    """Her touches as her HP number measured them (a drop from one read to the next, nothing unread between: exact),
    kept across runs in her own file (paths.of().touches), a JSON line each: t (the wall clock), map, level (as
    perception read it then: behavior.bootstrap; None: not read yet, counted at no level), hp, kind (kind_of_touch;
    None: none fits, counted for none; a row without one counts for none too), max (her max HP then), pads (the PADamage of the map's kinds then: what her DEF is fitted to,
    bootstrap.fit_defence). `path` None: kept in memory only."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.rows: list[dict] = []
        if path is not None:
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict):
                        self.rows.append(row)
            except OSError:  # (none kept yet)
                pass

    def measured(self, level: int | None) -> dict[str, tuple[int, int, int]]:
        """Her touches by kind as they'd be at `level`: (the largest, how many, the level it was taken at), of all
        those kept at `level` and her last CARRY_N kept at any other (none at no level). One taken at another level is
        carried to `level` by the formula's level term (`carried`): a level-up mustn't forget what the game hit her
        harder than the formula says (the review of 2026-10-28: a measured 130 over the formula's 122 at level 31 gone at
        32, the reserve back to the formula's until a touch of that kind was measured again), and her last CARRY_N are
        her gear as her DEF's fit has it (bootstrap.fit_defence's window): older ones of other levels go."""
        out: dict[str, tuple[int, int, int]] = {}
        if level is None:
            return out
        kept = [r for r in self.rows if isinstance(r.get("level"), (int, float)) and isinstance(r.get("hp"), (int, float))]
        recent = {id(r) for r in kept[-CARRY_N:]}
        for r in kept:
            k = r.get("kind")
            if k is None or r["level"] != level and id(r) not in recent:
                continue
            hp = carried(int(r["hp"]), r["level"], level)
            most, n, at = out.get(k, (0, 0, level))
            out[k] = (hp, n + 1, int(r["level"])) if hp > most else (most, n + 1, at)
        return out

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass


class HuntStop(Exception):
    """What ends a hunt for whoever is at the PC to put right, raised by Hunter.health_check: the safety stop, and out
    of darts (behavior.darts.OutOfDarts). Around every hunting command (behavior.retreat.on_alarm) she goes to a safe
    place as on the Lie Detector, the log says why in capitals (`label`: "<label>: <the stop>. The hunt ends: <ends>"),
    a telemetry event says so (`event`), whoever is at the PC is paged (`page`: a title and a line), the rolling
    recording's last `keep_s` s are saved (0: none), and the command exits with `exit_code`, printing `after` and
    `advice` (gamelab.__main__). `title`: its retreat's telemetry events' word (behavior.retreat.Why). `parks`: her
    retreat goes to the place a park would, however far (behavior.retreat.park_place; else STOP_S's best); `later`: no
    retreat now (the game isn't there), the guard's to do (behavior.guard)."""
    label = "STOP"
    title = "Stop"
    event = "stop"
    exit_code = 1
    keep_s = 0.0
    ends = "to a safe place"
    after = "The hunt ended; she went to a safe place (the log says where)."
    advice = ""
    at = 0.0
    parks = False
    later = False

    def page(self) -> tuple[str, str] | None:
        """What's paged (control.alarm.page): a title and a line; None: nothing."""
        return ("MapleStory: the hunt stopped", str(self))


class SafetyStop(HuntStop):
    """Her HP stayed under the safety stop's threshold: the hunt ends (raised by Hunter.health_check). `hp`: her HP at
    the look that ended it (HP; `exact`: her HP number's, else the bar's less a px); `seconds`: how long it had been
    under; `looks`: the looks of it; `threshold` (HP) and `why`, it in words ("3 Horny Mushroom touches of 129");
    `max_hp` (None: not read, `hp`, `threshold` and `floor` shares of her bar then); `at`: that look's time
    (time.monotonic); `floor`: the floor (HP) if it was under that which stopped it (FLOOR_LOOKS looks in a row), else
    None, and `floor_why` in words."""
    label = "SAFETY STOP"
    title = "Safety stop"
    event = "safety_stop"
    exit_code = EXIT
    ends = "to a safe place (no sitting, no waiting for HP)"
    advice = "Check her HP potions and her pet (fed, out, its potions) in the game before hunting again."

    def __init__(self, hp: float, seconds: float, looks: int, threshold: float, at: float = 0.0,
                 floor: float | None = None, *, max_hp: float | None = None, why: str = "", exact: bool = True,
                 floor_why: str = "the floor"):
        self.hp, self.seconds, self.looks, self.threshold, self.at = hp, seconds, looks, threshold, at
        self.floor, self.max_hp, self.why, self.exact, self.floor_why = floor, max_hp, why, exact, floor_why
        under = f"under {self.limit} for {seconds:.1f} s ({looks} looks)"
        if floor is not None:
            under += (f", under {floor_why} ({floor:.0f}) on {FLOOR_LOOKS} looks in a row" if max_hp else
                      f", under {floor_why} ({floor:.0%} of her bar) on {FLOOR_LOOKS} looks in a row")
        dead = hp <= (0.02 * max_hp if max_hp else 0.02)
        super().__init__(f"{self.her_hp}, {under}: her pet's potions aren't keeping her up (none left? the pet hungry "
                         "or not out?)" + ("; she may be dead" if dead else ""))

    @property
    def her_hp(self) -> str:
        """"HP 352 of 840" ("about" when the bar's; "about 41% of her bar" with her max HP not read)."""
        if not self.max_hp:
            return f"HP about {self.hp:.0%} of her bar"
        return f"HP {'' if self.exact else 'about '}{self.hp:.0f} of {self.max_hp:.0f}"

    @property
    def limit(self) -> str:
        """"3 Horny Mushroom touches of 129 (387)"."""
        th = f"{self.threshold:.0f}" if self.max_hp else f"{self.threshold:.0%}"
        return f"{self.why} ({th})" if self.why else th

    def page(self) -> tuple[str, str]:
        """What's paged (control.alarm.page): a title and a line."""
        return ("MapleStory: her HP stayed low",
                f"gamelab stopped the hunt: {self.her_hp}, under {self.limit} for {self.seconds:.1f} s. "
                "She went to a safe place. Check her potions and her pet.")


class StopAsked(HuntStop):
    """`python -m gamelab stop` (behavior.guard): the user's word, raised at the hunt's next look or glance
    (Hunter._stop_now): her moves alone from here, no page. `later`: the game wasn't back in behavior.guard.STOP_WAIT_S."""
    label = "STOP ASKED"
    title = "Stop asked"
    event = "stop_asked"
    exit_code = 9
    ends = "she parks"
    after = "The hunt ended as asked."
    parks = True

    def __init__(self, later: bool = False):
        self.later = later
        super().__init__("`gamelab stop`" + (": the game isn't in front, so the guard parks her at its return" if later
                                             else ""))

    def page(self) -> None:
        return None


class TooStrong(HuntStop):
    """The map's monsters are too strong for her: her reserve at her max HP is at or over her pet's warning (where her
    pet keeps her HP), so the safety stop would end the hunt at once. Raised before a hunting command hunts
    (gamelab.__main__'s _hunter: Hunter.too_strong), unless `--low-hp` sets the threshold."""
    label = "TOO STRONG"
    title = "Too strong"
    event = "too_strong"
    exit_code = EXIT
    ends = "nothing hunted"
    after = "Nothing was hunted."
    advice = "Hunt a map whose touches leave her pet's potions room, or give --low-hp to hunt here anyway."

    def __init__(self, threshold: float, max_hp: float, warning: float, why: str):
        """`threshold`: her reserve at her max HP `max_hp` (HP, read), `warning` her pet's (a share), `why` the reserve
        in words."""
        self.threshold, self.max_hp, self.warning, self.why = threshold, max_hp, warning, why
        super().__init__(f"the map's monsters are too strong for her here: {why} ({threshold:.0f} HP) is at or over "
                         f"her pet's warning, {warning:.0%} of her {max_hp:.0f} HP ({warning * max_hp:.0f}), where her "
                         "pet keeps her: the safety stop would end the hunt at once")

    def page(self) -> tuple[str, str]:
        return ("MapleStory: too strong for her here", str(self))


class HpNotRead(HuntStop):
    """Her max HP not read off her HUD's HP number by HP_WAIT_S into a hunting command (the perception service's
    newest states: gamelab.__main__'s _hunter, Hunter.wait_max_hp): nothing is hunted, unless `--low-hp` sets the
    threshold as a share of her bar (in HP it needs her max HP too). Without her max HP the safety stop can't count touches in HP, and no share of her bar stands for her
    profile's hits of the map's strongest touch on every map and at every level (the review of 2026-10-28: the 50% it
    hunted at was under 3 Horny Mushroom touches at levels 24-26, 52-55% of her HP then, and about 2 touches of a map
    whose largest is PADamage 150), nor can TooStrong be told."""
    label = "HP NOT READ"
    title = "HP not read"
    event = "hp_not_read"
    exit_code = EXIT
    ends = "nothing hunted"
    after = "Nothing was hunted."
    advice = ("Have the perception service (`gamelab perceive`) run on this code, with the game's client at 1366x768 "
              "and her HUD's HP number showing, or give --low-hp as a share of her HP bar (1 or under) to hunt with a "
              "threshold of your own.")

    def __init__(self, seconds: float, why: str = ""):
        self.seconds, self.why = seconds, why
        super().__init__(f"her max HP not read off her HUD's HP number in {seconds:g} s ({why or 'no state had it'}): "
                         "the safety stop can't count her profile's hits of the map's strongest touch in HP without it")

    def page(self) -> tuple[str, str]:
        return ("MapleStory: her HP number not read", str(self))


@dataclass
class HpWatch:
    """The safety stop's watch over her HP, a look at a time (`see`); `due` once it's to stop, then never again.
    `reserve`: hers on the map (the threshold, the floor); `potion_hp`: her own potion key's threshold (a share; None:
    no key of hers); `low_hp`: `--low-hp` (1 or under a share of her max HP, over 1 HP; None: the reserve's)."""
    reserve: Reserve = field(default_factory=Reserve)
    potion_hp: float | None = None
    low_hp: float | None = None
    max_hp: float | None = None  # her max HP as of the last look (the HUD's number's; None: not read, in shares then)
    mp: float | None = None      # ... and her MP (MP; None: not known), for Magic Guard's share
    since: float | None = None   # the first look of the stretch under the threshold (its time)
    looks: int = 0               # ... the looks of it
    low: int = 0                 # ... the looks in a row under the floor
    hp: float | None = None      # the last look's HP (HP)
    share: float | None = None   # ... its bar
    exact: bool = False          # ... whether her HP number's
    at: float = 0.0              # ... and its time
    fired: bool = False

    def threshold(self, max_hp: float | None = None, mp: float | None = None) -> float:
        """The threshold in HP at her max HP (the last look's by default); with her max HP not read, a share of her
        bar (the reserve's UNREAD_SHARE, `--low-hp`'s share; `--low-hp`'s HP can't be told then: 0)."""
        m = self.max_hp if max_hp is None else max_hp
        if self.low_hp is not None:
            if m is None:
                return self.low_hp if self.low_hp <= 1 else 0.0
            return self.low_hp * m if self.low_hp <= 1 else float(self.low_hp)
        th = self.reserve.hp(m, self.mp if mp is None else mp)
        if self.potion_hp is None:
            return th
        return min(th, (self.potion_hp - UNDER) * (1.0 if m is None else m))

    def floor(self, max_hp: float | None = None, mp: float | None = None) -> float:
        """The floor in HP: the reserve's, or the threshold if that's lower."""
        m = self.max_hp if max_hp is None else max_hp
        return min(self.reserve.floor(m, self.mp if mp is None else mp), self.threshold(m, mp))

    def words(self, max_hp: float | None = None, mp: float | None = None) -> str:
        """The threshold in words."""
        m = self.max_hp if max_hp is None else max_hp
        mp = self.mp if mp is None else mp
        of = "her bar" if m is None else f"{m:.0f}"
        if self.low_hp is not None:
            return f"--low-hp's {self.low_hp:.0%} of {of}" if self.low_hp <= 1 else "--low-hp's"
        if self.potion_hp is not None and (self.potion_hp - UNDER) * (1.0 if m is None else m) < self.reserve.hp(m, mp):
            return f"her potion key's {self.potion_hp - UNDER:.0%} of {of}"
        return self.reserve.words(m, mp)

    def floor_words(self, max_hp: float | None = None, mp: float | None = None) -> str:
        m = self.max_hp if max_hp is None else max_hp
        mp = self.mp if mp is None else mp
        return ("the threshold" if self.reserve.floor(m, mp) > self.threshold(m, mp)
                else self.reserve.floor_words(mp))

    def describe(self, max_hp: float | None = None, mp: float | None = None) -> str:
        """What it stops under, as the log says it."""
        m = self.max_hp if max_hp is None else max_hp
        mp = self.mp if mp is None else mp
        th = self.threshold(m, mp)
        if self.low_hp is not None or th < self.reserve.hp(m, mp):
            if m is None:
                return (f"under {th:.0%} of her bar ({self.words(m, mp)}; the reserve's {self.reserve.hp(m, mp):.0%}); "
                        f"at once under {self.floor(m, mp):.0%} two looks in a row")
            return (f"under {th:.0f} HP of {m:.0f} ({self.words(m, mp)}; the reserve's "
                    f"{self.reserve.hp(m, mp):.0f}); at once under {self.floor(m, mp):.0f} two looks in a row")
        return self.reserve.describe(m, mp)

    def see(self, t: float, share: float | None, max_hp: float | None, exact: float | None = None,
            mp: float | None = None, slack: float = BAR_SLACK) -> None:
        """A look at time `t` read her HP bar's share `share` (None: not read) and, if her HP number was read at its
        last change, `exact` (HP), with her max HP `max_hp` (None: not read, everything in shares of her bar then) and
        her MP `mp`. Neither: it neither starts nor ends a stretch."""
        if share is None and exact is None or t < self.at:
            return  # (nothing read; or a frame older than the last one seen: a look's after a glance's)
        if max_hp is None:
            if share is None:
                return
            exact = None  # (an HP number has its max: never here)
        self.max_hp, self.mp = max_hp, mp
        hp = float(exact) if exact is not None else max(0.0, share - slack) * (1.0 if max_hp is None else max_hp)
        self.hp, self.share, self.exact, self.at = hp, share, exact is not None, t
        self.low = self.low + 1 if hp < self.floor() else 0
        if hp >= self.threshold():
            self.since, self.looks = None, 0
            return
        if self.since is None:
            self.since = t
        self.looks += 1

    @property
    def debounce(self) -> float:
        """How long under the threshold stops it: DEBOUNCE_EXACT_S if the last look's HP was her HP number's and her pet
        drinks for her, else DEBOUNCE_S (her bar's; or a potion key of her own, `potion_hp`: her drink waits for a
        health check and her key, not the game's 0.04-0.20 s after a touch)."""
        return DEBOUNCE_EXACT_S if self.exact and self.potion_hp is None else DEBOUNCE_S

    def due(self) -> SafetyStop | None:
        """The stop, the first time her HP has been under the threshold on LOOKS looks DEBOUNCE_S apart at least, or
        under the floor FLOOR_LOOKS looks in a row; else None."""
        if self.fired or self.since is None:
            return None
        floor = self.low >= FLOOR_LOOKS
        if not floor and (self.looks < LOOKS or self.at - self.since < self.debounce):
            return None
        self.fired = True
        return SafetyStop(self.hp, self.at - self.since, self.looks, self.threshold(), self.at,
                          self.floor() if floor else None, max_hp=self.max_hp, why=self.words(), exact=self.exact,
                          floor_why=self.floor_words())
