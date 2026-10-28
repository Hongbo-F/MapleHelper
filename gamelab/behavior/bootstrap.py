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
  can't be counted in touches: the safety stop goes by her bar, under UNREAD_SHARE of it; the positioning model takes
  her max HP as PLAN_TOUCHES of the map's largest touch (more careful than she needs, on every map seen). Said.
- Her level: the HUD's "LV." digits (perception.lv_number), else her EXP number's (localization.gains: the EXP table
  puts its number at a level), the moment one reads; the digits' over the EXP number's when both have read. Until
  then a touch is worked out at UNREAD_LEVEL (a touch loses level / 10: at level 1 it takes the most). Said.
- Her DEF (her stat window's W.DEF): what the window says when it's read (perception.stat_window: prepared, not read
  live yet), kept in her stats file (paths.of().stats) and checked against her touches; else fitted to her own exact
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
UNREAD_SHARE = 0.5    # the safety stop's threshold while her max HP isn't read: this share of her bar (no HP to count
                      # touches in: under the 3 hits of 2026-10-28 by a third at DEF 0, under her pet's 70%)
PLAN_TOUCHES = 6.0    # the positioning model's max HP while hers isn't read: this many of the map's largest touch (a
                      # Horny Mushroom's 17%, where hers took 15-16% at levels 24-28: more careful, not less)
FIT_N = 40            # her last this many exact touches (of a known level, on a map whose monsters are known) ...
FIT_MIN = 8           # ... fit her DEF once this many are kept ...
FIT_AGREE = 0.8       # ... and at least this share of them fit one DEF (else none: said)
DEF_STEP = 0.25       # the DEFs tried (0 to DEF_TOP)
DEF_TOP = 400.0


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
            return f"DEF {span} (fitted to {self.agree} of her last {self.n} touches)"
        return (f"DEF not known ({self.why or 'no stat window read, no touches fitted'}): a touch at DEF "
                f"{self.low:g}, the most it takes")


PRIOR = Defence(PLAN_DEF, PRIOR_DEF, None, "prior")


def _taken(pad: float, d: np.ndarray, level: float, u: float) -> np.ndarray:
    """attack.danger.touch_taken over DEFs `d` at once (the same truncation and floor of 1)."""
    raw = pad * (1.2 + 0.2 * u)
    taken = raw * (1 - d / (d + 300 + 2 * raw)) - (d / 25 + level / 10)
    return np.maximum(1, np.trunc(taken))


def usable(row: dict) -> bool:
    """Whether a kept touch can be fitted: its size, its level (as perception read it then) and its map's monsters'
    PADamage ("pads")."""
    return (isinstance(row.get("hp"), (int, float)) and row.get("level") is not None and bool(row.get("pads")))


def fit_defence(rows, plan: float = PLAN_DEF) -> Defence:
    """Her DEF from her kept touches (behavior.safety.Touches.rows: those `usable`, the last FIT_N): at each DEF
    tried, how many of them some monster of their map could have taken by the formula at their level (anywhere in its
    roll); the DEFs the most fit, if FIT_AGREE of them or more do: `low` and `high` their ends, `best` the prior `plan`
    if it's among them, else their middle. Too few touches, or none agreeing: the prior, `why` saying so."""
    mine = [r for r in rows if usable(r)][-FIT_N:]
    n = len(mine)
    if n < FIT_MIN:
        return Defence(plan, PRIOR_DEF, None, "prior", n, why=f"{n} of the {FIT_MIN} touches to fit it kept, no stat "
                                                               "window read")
    d = np.arange(0.0, DEF_TOP + DEF_STEP / 2, DEF_STEP)
    fits = np.zeros(len(d), dtype=int)
    for r in mine:
        hp, level = float(r["hp"]), float(r["level"])
        ok = np.zeros(len(d), dtype=bool)
        for pad in r["pads"]:
            ok |= (_taken(pad, d, level, 0.0) <= hp) & (hp <= _taken(pad, d, level, TOP_U))
        fits += ok
    most = int(fits.max())
    if most < FIT_AGREE * n:
        return Defence(plan, PRIOR_DEF, None, "prior", n, most,
                       why=f"her last {n} touches don't agree on one (at most {most} fit any DEF)")
    best = d[fits == most]
    low, high = float(best.min()), float(best.max())
    middle = round((low + high) / 2 / DEF_STEP) * DEF_STEP
    return Defence(plan if low <= plan <= high else middle, low, high, "touches", n, most)


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

    def __init__(self, path: Path | None = None, rows=()):
        self.path = path
        self.max_hp: int | None = None
        self.level: int | None = None
        self.level_from = ""
        self.window: Defence | None = self._kept_window()
        self.defence: Defence = PRIOR
        self.refit(rows)
        self._disagree_said = False

    def _kept_window(self) -> Defence | None:
        """Her stat window's DEF as kept in her stats file, if it was read (never a fit's: those are made again)."""
        if self.path is None:
            return None
        try:
            kept = json.loads(self.path.read_text(encoding="utf-8")).get("window")
            return Defence(float(kept["low"]), float(kept["low"]), float(kept["low"]), "stat window",
                           at=kept.get("at"))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None

    def _save(self) -> None:
        if self.path is None:
            return
        out = {"defence": asdict(self.defence), "max_hp": self.max_hp, "level": self.level, "at": round(time.time(), 3)}
        if self.window is not None:
            out["window"] = {"low": self.window.low, "at": self.window.at}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        except OSError:
            pass

    @property
    def touch_level(self) -> int:
        """Her level for a touch: hers as read, else UNREAD_LEVEL."""
        return UNREAD_LEVEL if self.level is None else self.level

    def see_max_hp(self, most: int | None) -> str | None:
        if most is None or most <= 0 or most == self.max_hp:
            return None
        was, self.max_hp = self.max_hp, int(most)
        self._save()
        return f"her max HP {most} by {self.HP_FROM}" + ("" if was is None else f" (was {was})")

    def see_level(self, level: int | float | None, source: str) -> str | None:
        """Her level as `source` read it (LV_FROM or EXP_FROM): the digits' is taken over the EXP number's."""
        if level is None or level <= 0:
            return None
        level = int(level)
        if source == self.EXP_FROM and self.level_from == self.LV_FROM:
            if level != self.level and not self._disagree_said:
                self._disagree_said = True
                return f"her EXP number says level {level}, {self.LV_FROM} {self.level}: the digits' taken"
            return None
        if level == self.level and source == self.level_from:
            return None
        was, self.level, self.level_from = self.level, level, source
        if was == level:
            return None
        self._save()
        return f"her level {level} by {source}" + ("" if was is None else f" (was {was})")

    def see_window(self, wdef: float, at: float | None = None, rows=()) -> str | None:
        """Her stat window read her W.DEF `wdef` (perception.stat_window): hers from now on, kept, checked against her
        touches (said if they don't fit it: her measured touches raise her reserve past it anyway)."""
        self.window = Defence(float(wdef), float(wdef), float(wdef), "stat window", at=time.time() if at is None else at)
        return self.refit(rows)

    def refit(self, rows) -> str | None:
        """Her DEF again: her stat window's if read, else fitted to her touches (`rows`), else the prior. What changed,
        in words (None: nothing)."""
        was = self.defence
        if self.window is not None:
            ok, n = agreeing(self.window.low, rows)
            now = Defence(self.window.low, self.window.low, self.window.low, "stat window", n, ok, self.window.at)
        else:
            now = fit_defence(rows)
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
