"""Her stats from perception (the bootstrap): her max HP, her level and her DEF, which the safety stop's reserve
(behavior.safety) and the positioning model (attack.danger) are worked out from, each with where it came from, and
what's taken, said, until it's read. None of them is config.toml's or her profile's any more.

The user, 2026-10-28 ~15:30, on config.toml's [character] hp = 774 / level = 25 (her HUD said 840 at level 28 that day)
and the danger model's DEF 75: "These should not even be in config yaml, use perception result to bootstraps"; and at
~15:40: "I thought hp and level will come from perception, please remove them completely from config.toml", and "Why
not using the extracted monster stats (attack, etc) and hit damage formula (extracted or researched online) and the
character's defense, to estimate the hit damage?". So, in `HerStats`:
- Her max HP: the HUD's HP number ("HP [cur/max]": perception.hp_number, which the perception service's world step
  reads at every change of it; the feed's newest state at the hunter's start), the moment it reads. Until then her HP
  can't be counted in touches: a command that hunts waits for it at its start and hunts nothing without it
  (behavior.safety.HpNotRead: no share of her bar is her hits of the map's strongest touch on every map and level);
  elsewhere (a `nav` trip, a hunter whose eyes draw no number) the safety stop goes by her bar, under UNREAD_SHARE of
  it; the positioning model takes her max HP as PLAN_TOUCHES of the map's largest touch (more careful than she needs,
  on every map seen). Said.
- Her level: the HUD's "LV." digits (perception.lv_number), else her EXP number's (localization.gains: the EXP table
  puts its number at a level), the moment one reads; the digits' over the EXP number's while they read, the EXP
  number's once they haven't for LV_UNREAD_S. Until then a touch is worked out at UNREAD_LEVEL (a touch loses level /
  10: at level 1 it takes the most). Said.
- Her DEF (her stat window's W.DEF): what the window says when it's read (perception.stat_window, opened, read and
  closed at a hunt's start by behavior.statcheck: the user, 2026-10-28 ~18:20, "You can open character stats panel to
  check the defense"), kept in her stats file (paths.of().stats) with the level and the session it was read in, and
  checked against her touches (`window_due`: read again at another level, in another session, or once her newest
  touches since don't fit it); else fitted to her own exact
  touches (`fit_defence`: attack.danger.touch_taken, the game's formula, at the level each was taken at, over the
  PADamage of the map's monsters then: the DEF most of her last FIT_N fit, once FIT_MIN are kept, refitted at every
  touch); else none known: the safety stop takes a touch at PRIOR_DEF (0, the most a touch can take), the positioning
  model at PLAN_DEF (the stack's prior, attack.danger.DEFENCE). Said, and a fit kept in her stats file too.

The fit, on her 178 exact touches of 2026-10-24 to 28 (her HUD's HP number off the 17 recordings:
D:/MapleHelper-scratch/safety3/design; tests/fixtures/touches_2026-10-28.jsonl, tests/test_bootstrap.py): DEF 48-59.5
at level 24 (15 of 15 touches fit), 59.75-64.75 at 25 (40 of 40), 61.5-70.25 at 26 (35 of 35), 106.5-115.25 at 28 (40
of 40; a Horny Mushroom 103-122, a Green 44-52): new gear between 10-26 23:47 and 10-28 00:30, which no number in a file
would have followed. A fit takes the lowest DEF the most of her touches allow for the safety stop (`Defence.low`: the
most a touch can take of those), and for the positioning model the prior while it's in what they allow (no change,
nothing replanned), else their middle (`Defence.best`; the hunter rounds it to 5).
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..attack import danger
from ..attack.danger import DEFENCE, TOP_U, touch_taken

PRIOR_DEF = 0.0       # the safety stop's DEF while hers isn't read or fitted: none (a touch's most)
PLAN_DEF = DEFENCE    # ... the positioning model's (attack.danger): the stack's prior, kept while her touches allow it
UNREAD_LEVEL = 1      # her level for a touch while it isn't read (a touch loses level / 10: at 1 it takes the most)
UNREAD_SHARE = 0.5    # the safety stop's threshold while her max HP isn't read, where nothing hunts (a `nav` trip, a
                      # hunter whose eyes draw no HP number): this share of her bar, under her pet's 70%. No share is
                      # her hits of the map's strongest touch everywhere (3 Horny Mushroom touches were 52-55% of her
                      # HP at levels 24-26), so a command that hunts hunts nothing without it (safety.HpNotRead)
PLAN_TOUCHES = 6.0    # the positioning model's max HP while hers isn't read: this many of the map's largest touch (a
                      # Horny Mushroom's 17%, where hers took 15-16% at levels 24-28: more careful, not less)
FIT_N = 40            # her last this many exact touches (of a known level, on a map whose monsters are known) ...
FIT_MIN = 8           # ... fit her DEF once this many are kept ...
FIT_AGREE = 0.8       # ... and at least this share of them fit one DEF (else none: said); her newest FIT_MIN of them
                      # fit it too, this share of them, else her DEF may have changed (`fit_defence`)
DEF_STEP = 0.25       # the DEFs tried (0 to DEF_TOP)
DEF_TOP = 400.0
LV_UNREAD_S = 2.0     # s: her LV. digits not read this long (the box covered, frames failing their checks), her EXP
                      # number's level is taken over what they last said (`HerStats.see_level`)
SESSION_S = 3 * 3600.0  # s: her stat window's read is this session's for this long at the most (`window_due`; a session
                        # of the perception service's another's at once): she's played by hand between the bot's runs
                        # (new gear on 10-26 to 10-28), so an evening's read isn't the next one's
# What her stat window shows besides her W.DEF, kept with it in her stats file (`see_window`'s `read`): her level, max HP
# and MP, ATTACK, ACCURACY and EVASION (her MISS chance and a touch's: not modelled from them yet), MAGIC, MAGIC DEF., her
# four stats, SPEED (her walk and climb's, where it's taken: `see_window`'s `speed`, behavior.hunter) and JUMP (her
# profile's plans her jumps: said where they differ, statcheck.moves_notes)
WINDOW_KEEPS = ("level", "hp", "mp", "attack", "acc", "eva", "magic", "mdef", "str", "dex", "int", "luk", "speed",
                "jump")


@dataclass(frozen=True)
class Defence:
    """Her DEF as known: `best` for the positioning model (attack.danger), `low` for the safety stop's touches (the
    lowest the evidence allows: the most a touch can take of it), `high` the highest (None: no bound); `source`:
    "stat window" (read off it), "touches" (fitted to `agree` of her last `n`), "prior" (neither: `why`); `at`: when
    the window was read (time.time; None)."""
    best: float
    low: float
    high: float | None = None
    source: str = "prior"
    n: int = 0
    agree: int = 0
    at: float | None = None
    why: str = ""

    def words(self) -> str:
        """Where it comes from, in words: "DEF about 111 (110.5-111.25, fitted to 38 of her last 40 touches)"."""
        if self.source == "stat window":
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.at)) if self.at else "?"
            return f"DEF {self.low:g} (her stat window's W.DEF, read {when})"
        if self.source == "touches":
            span = f"{self.low:g}-{self.high:g}" if self.high is not None and self.high > self.low else f"{self.low:g}"
            return f"DEF {span} (fitted to {self.agree} of her last {self.n} touches{'; ' + self.why if self.why else ''})"
        return (f"DEF not known ({self.why or 'no stat window read, no touches fitted'}): a touch at DEF "
                f"{self.low:g}, the most it takes")


PRIOR = Defence(PLAN_DEF, PRIOR_DEF, None, "prior")


def session_of_feed(feed) -> str | None:
    """The perception service's run a feed's looks come from (its welcome's pid: a session of the game's, as near as
    it's known), None without one (the simulator, a test's feed, no feed)."""
    w = getattr(feed, "welcome", None)
    return None if not isinstance(w, dict) or w.get("pid") is None else str(w["pid"])


def _taken(pad: float, d: np.ndarray, level: float, u: float) -> np.ndarray:
    """attack.danger.touch_taken over DEFs `d` at once (the same truncation and floor of 1)."""
    raw = pad * (1.2 + 0.2 * u)
    taken = raw * (1 - d / (d + 300 + 2 * raw)) - (d / 25 + level / 10)
    return np.maximum(1, np.trunc(taken))


def usable(row: dict) -> bool:
    """Whether a kept touch can be fitted: its size, its level (as perception read it then) and its map's monsters'
    PADamage ("pads")."""
    return (isinstance(row.get("hp"), (int, float)) and row.get("level") is not None and bool(row.get("pads")))


def _fit_rows(mine, d: np.ndarray) -> np.ndarray:
    """Which of DEFs `d` each touch of `mine` fits (a row each): some monster of its map could have taken it by the
    formula at its level, anywhere in its roll."""
    out = np.zeros((len(mine), len(d)), dtype=bool)
    for i, r in enumerate(mine):
        hp, level = float(r["hp"]), float(r["level"])
        for pad in r["pads"]:
            out[i] |= (_taken(pad, d, level, 0.0) <= hp) & (hp <= _taken(pad, d, level, TOP_U))
    return out


def _ends(fits: np.ndarray, d: np.ndarray) -> tuple[float, float, int]:
    """The DEFs the most of `fits`' touches fit: their lowest, their highest, and how many fit them."""
    count = fits.sum(axis=0)
    most = int(count.max())
    best = d[count == most]
    return float(best.min()), float(best.max()), most


def fit_defence(rows, plan: float = PLAN_DEF) -> Defence:
    """Her DEF from her kept touches (behavior.safety.Touches.rows: those `usable`, the last FIT_N): at each DEF
    tried, how many of them some monster of their map could have taken by the formula at their level (anywhere in its
    roll); the DEFs the most fit, if FIT_AGREE of them or more do: `low` and `high` their ends, `best` the prior `plan`
    if it's among them, else their middle. Too few touches, or none agreeing: the prior, `why` saying so.

    Her newest FIT_MIN are checked against it: a DEF that went down (gear taken off) leaves the last FIT_N agreeing on
    the old one for touches after (the review of 2026-10-28, her 178 touches in reverse: 9 touches at the old low end,
    then 91.25, 77.5, 73.75, at 69.25 only about 32 in, a Horny Mushroom's top 124-131 meanwhile against the 134 she
    could take), and her reserve under her touches all that time. So where under FIT_AGREE of the newest fit a DEF in
    the fit's span, their own fit's span is taken with it (the lower low: the most a touch takes of either), said in
    `why`; and where they don't agree on one either, the prior (DEF 0), said, until they do. In reverse now: DEF 0 for
    the 4 touches after the drop, 69.25 at the 5th, 59.25 at the 7th (the level's own fit 61.5-70.25); a Horny
    Mushroom's top under the 134 she could take at 3 touches, by 4 HP at the most, against 23 by up to 10
    (D:/MapleHelper-scratch/safety3/stats-fix/fit_deficit.py). A DEF that went up (new gear: 10-26 to 10-28) moves the
    low end no higher than the fit of the last FIT_N does: in time order the fit is the same at every touch but the 4
    at the new gear's start, at DEF 0 (fit_timeorder.py)."""
    mine = [r for r in rows if usable(r)][-FIT_N:]
    n = len(mine)
    if n < FIT_MIN:
        return Defence(plan, PRIOR_DEF, None, "prior", n, why=f"{n} of the {FIT_MIN} touches to fit it kept, no stat "
                                                               "window read")
    d = np.arange(0.0, DEF_TOP + DEF_STEP / 2, DEF_STEP)
    fits = _fit_rows(mine, d)
    low, high, most = _ends(fits, d)
    if most < FIT_AGREE * n:
        return Defence(plan, PRIOR_DEF, None, "prior", n, most,
                       why=f"her last {n} touches don't agree on one (at most {most} fit any DEF)")
    why = ""
    newest = fits[-FIT_MIN:]
    ok = int(newest[:, (d >= low) & (d <= high)].any(axis=1).sum())
    if ok < FIT_AGREE * FIT_MIN:
        new_low, new_high, agree = _ends(newest, d)
        span = f"{low:g}-{high:g}" if high > low else f"{low:g}"
        if agree < FIT_AGREE * FIT_MIN:
            return Defence(plan, PRIOR_DEF, None, "prior", n, most,
                           why=f"her newest {FIT_MIN} touches fit neither the DEF her last {n} do ({span}: {ok} of "
                               f"them) nor one of their own (at most {agree}): her gear changed?")
        why = (f"her newest {FIT_MIN} fit {new_low:g}-{new_high:g}, {ok} of them her last {n}'s {span}: the lower "
               "taken")
        low, high = min(low, new_low), max(high, new_high)
    middle = round((low + high) / 2 / DEF_STEP) * DEF_STEP
    return Defence(plan if low <= plan <= high else middle, low, high, "touches", n, most, why=why)


def agreeing(defence: float, rows) -> tuple[int, int]:
    """How many of her last FIT_N usable touches the formula gives at `defence`, of how many (a stat window's DEF checked
    against them)."""
    mine = [r for r in rows if usable(r)][-FIT_N:]
    ok = sum(any(touch_taken(p, defence, r["level"], 0.0) <= r["hp"] <= touch_taken(p, defence, r["level"], TOP_U)
                 for p in r["pads"]) for r in mine)
    return ok, len(mine)


class HerStats:
    """Her stats as perception has read them this run (see the top): `max_hp` (None: not yet), `level` (None: not
    yet) and where it came from (`level_from`), `defence` (a Defence). Each `see_*` returns what changed in words (None:
    nothing), for whoever logs it and makes her reserve again (Hunter._reserve_check). `path`: her stats file (her
    stat window's DEF read, kept; the DEF last fitted, for `gamelab character`), None: none."""

    HP_FROM = "her HUD's HP number"
    LV_FROM = "her HUD's LV. digits"
    EXP_FROM = "her EXP number"
    WINDOW_FROM = "her stat window"

    def __init__(self, path: Path | None = None, rows=(), session: str | None = None):
        self.path = path
        self.max_hp: int | None = None
        self.level: int | None = None
        self.level_from = ""
        # This run, for judging a W.DEF kept from before it (`window_doubt`): when it started, the perception service's
        # session it looks through (`session`: its pid, session_of_feed; None: not known), her first level read in it
        self.run_start = time.time()
        self.session = None if session is None else str(session)
        self.first_level: int | None = None
        self._rows = rows
        self._read: dict[str, float] = {}  # when this run read her max HP and her level (time.time), by key
        self._kept = self._kept_file()     # her stats file as this run found it
        self._disagree: tuple | None = None  # (the EXP number's level, the digits') last said to disagree
        self._lv_unread: float | None = None  # her LV. digits not read since (see_level's t), once they had read
        self._lv_gone = False                 # ... and not read since they last did
        self.window: Defence | None = self._kept_window()
        self.window_read: dict = self._kept_read()  # what else the window showed (WINDOW_KEEPS), its level and session
        self.defence: Defence = PRIOR
        self.refit(rows)

    def _kept_file(self) -> dict:
        if self.path is None:
            return {}
        try:
            kept = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return kept if isinstance(kept, dict) else {}

    def _kept_window(self) -> Defence | None:
        """Her stat window's DEF as kept in her stats file, if it was read (never a fit's: those are made again)."""
        try:
            kept = self._kept.get("window")
            return Defence(float(kept["low"]), float(kept["low"]), float(kept["low"]), "stat window",
                           at=kept.get("at"))
        except (ValueError, TypeError, KeyError, AttributeError):
            return None

    @property
    def kept_max_hp(self) -> int | None:
        """Her max HP as her stats file kept it (her HUD's number, last read in a run before this one), None if none."""
        v = self._kept.get("max_hp")
        return int(v) if isinstance(v, (int, float)) and v > 0 else None

    def _kept_read(self) -> dict:
        """The rest of her stat window's read as kept (its level, session and WINDOW_KEEPS' values, and the SPEED her
        walk and climb were set to from her stat window then), {} if none."""
        kept = self._kept.get("window")
        if self.window is None or not isinstance(kept, dict):
            return {}
        return {k: kept[k] for k in ("level", "session", "read", "speed") if k in kept}

    def _save(self) -> None:
        """Her stats file: her DEF as known now (and her stat window's, if read), and her max HP and level as last read,
        each with when: this run's once it has read it, else as the file had it (a run that reads neither, or ends
        before it does, leaves them as they were: `gamelab stats` and an offline `patrol` read them)."""
        if self.path is None:
            return
        out = {"defence": asdict(self.defence)}
        for key, mine in (("max_hp", self.max_hp), ("level", self.level)):
            if key in self._read:
                out[key], out[f"{key}_at"] = mine, round(self._read[key], 3)
            else:
                out[key], out[f"{key}_at"] = self._kept.get(key), self._kept.get(f"{key}_at", self._kept.get("at"))
        if "level" in self._read:
            out["level_from"] = self.level_from
        elif "level_from" in self._kept:
            out["level_from"] = self._kept["level_from"]
        out["at"] = round(time.time(), 3)
        if self.window is not None:
            out["window"] = {"low": self.window.low, "at": self.window.at, **self.window_read}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        except OSError:
            pass

    @property
    def touch_level(self) -> int:
        """Her level for a touch: hers as read, else UNREAD_LEVEL."""
        return UNREAD_LEVEL if self.level is None else self.level

    def see_max_hp(self, most: int | None, source: str | None = None) -> str | None:
        """Her max HP as `source` read it (HP_FROM by default; WINDOW_FROM: her stat window's, taken only while her
        HUD's number hasn't read)."""
        if most is None or most <= 0 or most == self.max_hp:
            return None
        was, self.max_hp = self.max_hp, int(most)
        self._read["max_hp"] = time.time()
        self._save()
        return f"her max HP {most} by {source or self.HP_FROM}" + ("" if was is None else f" (was {was})")

    def see_level(self, level: int | float | None, source: str, t: float | None = None) -> str | None:
        """Her level as `source` read it (LV_FROM or EXP_FROM) at time `t` (a look's or glance's; None: not timed):
        the digits' over the EXP number's while they read; once they haven't for LV_UNREAD_S (a None from LV_FROM
        says they didn't: untimed, at once), the EXP number's, said, until they read again. A disagreement between the
        two is said once for each pair of levels. Her first level this run judges a W.DEF kept from before it
        (`window_doubt`: read at another level, not hers): her DEF made again then, said with it."""
        note = self._see_level(level, source, t)
        if self.first_level is None and self.level is not None:
            self.first_level = self.level
            if self.window is not None and self._before_run(self.window):
                more = self.refit(self._rows)
                if more is not None:
                    note = more if note is None else f"{note}; {more}"
        return note

    def _see_level(self, level: int | float | None, source: str, t: float | None = None) -> str | None:
        if source == self.LV_FROM:
            if level is None or level <= 0:
                if self.level_from == self.LV_FROM and not self._lv_gone:
                    self._lv_gone, self._lv_unread = True, t
                return None
            self._lv_gone, self._lv_unread = False, None
        elif level is None or level <= 0:
            return None
        level = int(level)
        if source == self.EXP_FROM and self.level_from == self.LV_FROM:
            stale = self._lv_gone and (t is None or self._lv_unread is None or t - self._lv_unread >= LV_UNREAD_S)
            if not stale:
                if level != self.level and self._disagree != (level, self.level):
                    self._disagree = (level, self.level)
                    return f"her EXP number says level {level}, {self.LV_FROM} {self.level}: the digits' taken"
                return None
            if level == self.level:
                return None
            was, self.level, self.level_from = self.level, level, source
            self._read["level"] = time.time()
            self._save()
            return (f"her level {level} by {source} ({self.LV_FROM} not read for {LV_UNREAD_S:g} s or more; they said "
                    f"{was})")
        if level == self.level and source == self.level_from:
            return None
        was, self.level, self.level_from = self.level, level, source
        self._read["level"] = time.time()
        if was == level:
            return None
        self._save()
        return f"her level {level} by {source}" + ("" if was is None else f" (was {was})")

    def see_window(self, wdef: float, at: float | None = None, rows=None, level: int | None = None,
                   session: str | None = None, read: dict | None = None, speed: float | None = None) -> str | None:
        """Her stat window read her W.DEF `wdef` (perception.stat_window; behavior.statcheck opens it): hers from now
        on, kept with her level then (`level`: the window's own, else hers as read; None: not known), the session it was
        read in (`session`: the perception service's run; None: none), what else it showed (`read`: WINDOW_KEEPS') and
        the SPEED her walk and climb are set to from her stat window (`speed`: its read's, or one before where a read's
        isn't taken, behavior.hunter; None: none taken, her profile's), checked against her touches (said if they don't
        fit it: her measured touches raise her reserve past it anyway).
        What changed, in words; None if her DEF is as it was (a read again: kept with its new time). A W.DEF outside 0 to
        DEF_TOP is no read (a misread: the review of 2026-10-28, an OCR's "711" for 111 put her reserve at 117 HP of
        840): not kept, her DEF as it was, said. One her touches don't bear out isn't her DEF either (`window_doubt`)."""
        if not 0 <= float(wdef) <= DEF_TOP:
            return (f"her stat window's W.DEF {wdef:g} not taken: outside 0-{DEF_TOP:g} (a misread?); her DEF as it "
                    f"was, {self.defence.words()}")
        when = time.time() if at is None else at
        self.window = Defence(float(wdef), float(wdef), float(wdef), "stat window", at=when)
        level = level if level is not None else self.level
        self.window_read = {"level": level, "session": session,
                            "read": {k: (list(v) if isinstance(v, tuple) else v) for k, v in (read or {}).items()
                                     if k in WINDOW_KEEPS and v is not None},
                            **({} if speed is None else {"speed": float(speed)})}
        rows = self._rows if rows is None else rows  # (her touches as last refitted: the hunter's)
        note = self.refit(rows)
        if note is None:
            self._save()  # (the same DEF read again: when, at what level, kept)
            if self.defence.source != "stat window":  # (not borne out by her touches: her DEF as it was, said)
                note = f"her stat window's W.DEF {wdef:g} not taken ({self.window_doubt(rows)}): her DEF as it was"
        return note

    def window_due(self, level: int | None = None, session: str | None = None, rows=None,
                   now: float | None = None) -> str | None:
        """Why her stat window should be read now (None: her DEF is known from it this session): none read; a doubt of
        the read kept (`window_doubt`, with `session` and `level` for this run's and `now` for its start: read in
        another session, at another level, over SESSION_S before, her newest touches since not fitting it, over what
        her touches fit); or read at another level than `level` (hers now: after a level-up in the hunt, read again,
        her DEF kept meanwhile)."""
        if self.window is None:
            return "no stat window read yet"
        if (why := self.window_doubt(rows, session=session, level=level, now=now)) is not None:
            return why
        read_level = self.window_read.get("level")
        if level is not None and read_level is not None and int(read_level) != int(level):
            return f"her W.DEF {self.window.low:g} read at level {read_level}; she's level {level} now"
        return None

    def _before_run(self, w: Defence, now: float | None = None) -> bool:
        """Whether W.DEF `w` was read before this run (`now`: its start, run_start, by default)."""
        return w.at is None or w.at < (self.run_start if now is None else now)

    def window_doubt(self, rows=None, session: str | None = None, level: int | None = None,
                     now: float | None = None) -> str | None:
        """Why her W.DEF as read isn't her DEF now (None: it is; `refit`). A read from before this run (the review of
        2026-10-28: a stale read kept governing whenever the read again didn't happen: a hunt put off at its start, six
        put-offs, no calibration, her stats key opening nothing): read in another session of the perception service
        than this run's (`session`, else `self.session`), at another level than her first this run (`level`, else
        `first_level`), or over SESSION_S before it started (`now`, else `run_start`): she plays by hand between runs.
        Any read: her newest FIT_MIN touches since it not fitting it (under FIT_AGREE: her gear changed, or a misread);
        or, with fewer than FIT_MIN since to bear it out, over the DEF her last FIT_N touches fit (`fit_defence`'s span,
        FIT_AGREE of them not fitting it: a misread, or new gear her touches since will show). A W.DEF under what her
        touches fit isn't doubted: the lower is the safe side (a touch takes more of her), and gear taken off shows so."""
        w = self.window
        if w is None:
            return None
        rows = self._rows if rows is None else rows
        if self._before_run(w, now):
            read_level, lv = self.window_read.get("level"), self.first_level if level is None else level
            if lv is not None and read_level is not None and int(read_level) != int(lv):
                return f"her W.DEF {w.low:g} read at level {read_level}; she's level {lv} now"
            kept, session = self.window_read.get("session"), self.session if session is None else session
            if session is not None and kept is not None and str(kept) != str(session):
                return (f"her W.DEF {w.low:g} read in another session (the perception service's {kept}; now "
                        f"{session})")
            start = self.run_start if now is None else now
            if w.at is None or start - w.at > SESSION_S:
                ago = "?" if w.at is None else f"{(start - w.at) / 3600:.1f} h"
                return f"her W.DEF {w.low:g} read {ago} ago (over {SESSION_S / 3600:g} h)"
        since = [r for r in rows if usable(r) and isinstance(r.get("t"), (int, float)) and r["t"] >= (w.at or 0.0)]
        if len(since) >= FIT_MIN:
            ok, n = agreeing(w.low, since[-FIT_MIN:])
            if ok < FIT_AGREE * n:
                return (f"only {ok} of her newest {n} touches since it was read fit her W.DEF {w.low:g} (her gear "
                        "changed?)")
            return None
        fit = fit_defence(rows)
        if fit.source == "touches" and fit.high is not None and w.low > fit.high:
            ok, n = agreeing(w.low, rows)
            if ok < FIT_AGREE * n:
                span = f"{fit.low:g}-{fit.high:g}" if fit.high > fit.low else f"{fit.low:g}"
                return (f"her W.DEF {w.low:g} over the DEF her last {n} touches fit ({span}; {ok} of them fit "
                        f"{w.low:g}), {len(since)} of the {FIT_MIN} touches since the read to bear it out (a misread, "
                        "or new gear?)")
        return None

    def refit(self, rows) -> str | None:
        """Her DEF again: her stat window's if read and not doubted (`window_doubt`), else fitted to her touches
        (`rows`), else the prior (a doubted W.DEF said in its `why`). What changed, in words (None: nothing)."""
        was = self.defence
        self._rows = rows
        doubt = self.window_doubt(rows) if self.window is not None else None
        if self.window is not None and doubt is None:
            ok, n = agreeing(self.window.low, rows)
            now = Defence(self.window.low, self.window.low, self.window.low, "stat window", n, ok, self.window.at)
        else:
            now = fit_defence(rows)
            if doubt is not None:
                if now.source == "touches":
                    why = f"her stat window's not taken: {doubt}" + (f"; {now.why}" if now.why else "")
                else:
                    why = (f"her stat window's not taken: {doubt}; "
                           + (now.why.replace(", no stat window read", "") if now.why else "no touches fitted"))
                now = Defence(now.best, now.low, now.high, now.source, now.n, now.agree, now.at, why)
        self.defence = now
        if (now.best, now.low, now.high, now.source) == (was.best, was.low, was.high, was.source):
            return None
        self._save()
        text = f"her {now.words()}"
        if now.source == "stat window" and now.n and now.agree < FIT_AGREE * now.n:
            text += (f"; only {now.agree} of her last {now.n} touches fit it (her measured touches raise her reserve "
                     "past it)")
        if was.source != "prior" or now.source != "prior":
            text += f" (was {was.words()})" if was.source != now.source else ""
        return text

    def her(self, largest_touch: float = 1.0) -> danger.Her:
        """What the positioning model measures a touch against (attack.danger.Her): her max HP as read, else
        PLAN_TOUCHES of the map's `largest_touch` (HP: at her level and DEF as known), her level as read, else
        UNREAD_LEVEL, and her DEF's `best`."""
        hp = self.max_hp if self.max_hp is not None else PLAN_TOUCHES * max(1.0, largest_touch)
        return danger.Her(float(hp), float(self.touch_level), float(self.defence.best))

    def describe(self) -> str:
        """Each stat and where it came from, as the log says it."""
        hp = (f"max HP {self.max_hp} ({self.HP_FROM})" if self.max_hp is not None
              else f"max HP not read yet ({self.HP_FROM})")
        lv = (f"level {self.level} ({self.level_from})" if self.level is not None
              else f"level not read yet (a touch at level {UNREAD_LEVEL}, the most it takes)")
        return f"her {hp}; {lv}; {self.defence.words()}"
