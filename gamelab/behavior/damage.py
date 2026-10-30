"""Her attacks' damage as the monsters' HP bars show it: learned as she hunts and kept between runs in her own file
(paths.of().damage), so what she reckons a monster has left (behavior.attacks, attack.choice.Left) goes by her damage as
it is, not by a prior. The user, 2026-10-29, after the bars of that night's grinds showed Magic Claw's prior far off
(two hits of 76-267 each, the v83 formula on her stat window; the bars a cast's two at 254-298 before the defence, p10
to p90, so a Green Mushroom, 303 before its MDD of 30, came out of one cast with about a tenth left nearly every time,
where the prior had 69% of the casts that land kill it, and right after a cast reckoned it 50% dead: her second cast
held back as 'likely dead'): "The fitted damage should be dynamically updated as grinding goes, and memorized between
runs".

What she learns from (behavior.attacks.Attacks, look by look: `Attacks._learn_bar`, `_learn_kill`):
- a monster's bar read between empty and full after exactly one throw of hers whose damage can have shown since the bar
  before (or since her first throw at it, no bar up then: full), and none whose damage may or may not have shown by
  either read: that throw's damage, the bar's drop x its HP before its defence (x (100 + defence) / 100), 'exact' give
  or take a px of the bar's fill (BAR_PX); a drop of under MISS_PX px after a bar before is a miss (it took nothing);
- a kill of hers placed on it after exactly one throw so: that throw did at least what it had left ('at least').
A bar shows the damage of whoever dealt it, and another player's isn't told from hers, so nothing is learned at a
monster with another player on its platform or near it (Attacks: OTHERS_DX).

The fit (`fit`): a hit's damage a whole number from lo to hi, each as likely (attack.choice.Attack's model), `hits` of
them a throw: the most likely (lo, hi) on a grid over the kept reads, any one of them anything at all with the chance
OUTLIER (another's damage unseen, a misread bar). Used from MIN_EXACT exact reads on (`Book.attack`), the profile's prior
until then; her misses' share from MIN_MISSES judged throws on (`Book.miss`). The last KEEP reads of each attack are
kept, those of her level and LEVELS_BACK under it used (her damage rises with her INT as she levels)."""

import atexit
import json
import math
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..attack.choice import Attack

BAR_PX = 46        # px of a monster's HP bar's fill (perception.hud.BAR_FILL): a read's give or take, a px
TOL_MIN = 3.0      # damage: a read's least give or take
MISS_PX = 0.5      # px: a bar after one throw lower than the one before by less than this: the throw took nothing
EXACT, AT_LEAST, MISS = "exact", "at_least", "miss"
OUTLIER = 0.05     # the chance a read is anything at all (another player's damage unseen, a misread bar) ...
OUTLIER_KILL = 0.15  # ... a kill's (one placed on her mark that was another's, or one another player had hurt unseen)
KILL_SLACK = 0.15  # a kill's 'at least' over the most an exact read had by more than this share: not hers, not used
MIN_EXACT = 15     # exact reads before the fit is used (the profile's prior until then)
MIN_MISSES = 20    # throws judged hit or miss before her misses' share is (the profile's MISS until then)
KEEP = 400         # reads kept an attack (the newest)
LEVELS_BACK = 1    # reads of her level and this many under it are used
REFIT_S = 10.0     # s between fits at most (one takes 0.1-0.4 s, on a thread of its own: `Book.version`)
SAVE_S = 60.0      # s between saves at most (and at the run's end: `save`, atexit)
GRID = (48, 64)    # the fit's grid: lo's steps, hi's steps (then a finer one about the best)
THREADED = True    # the fits on a thread of their own (tests: at once, the same every run)


@dataclass(frozen=True)
class Read:
    """One thing a throw of hers showed: `how` EXACT (its damage `value`, give or take `tol`), AT_LEAST (it killed: its
    damage at least `value`) or MISS (it took nothing); its damage before the monster's defence. When (wall clock), at
    which kind, at her level then (None: not read), and whether a bar was read before it (`barred`: a miss could have
    shown; from no bar up a miss leaves none, which says nothing)."""
    t: float
    kind: str
    how: str
    value: float
    tol: float
    level: int | None = None
    barred: bool = False

    def row(self) -> list:
        return [round(self.t, 2), self.kind, self.how, round(self.value, 1), round(self.tol, 1), self.level,
                int(self.barred)]

    @classmethod
    def of(cls, row: list) -> "Read":
        t, kind, how, value, tol, level, *rest = row
        return cls(float(t), str(kind), str(how), float(value), float(tol), None if level is None else int(level),
                   bool(rest[0]) if rest else False)


def _cdf(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, hits: int) -> np.ndarray:
    """P(a throw's damage <= x) for `hits` hits each a whole number from lo to hi as likely each (as continuous on
    lo - 0.5 to hi + 0.5): one hit's uniform, two a triangle, more as the normal of their sum."""
    a, b = lo - 0.5, hi + 0.5
    span = b - a
    if hits <= 1:
        return np.clip((x - a) / span, 0.0, 1.0)
    if hits == 2:
        s = np.clip(x - 2 * a, 0.0, 2 * span)
        return np.where(s <= span, s * s / (2 * span * span), 1.0 - (2 * span - s) ** 2 / (2 * span * span))
    mean, sd = hits * (a + b) / 2, np.sqrt(hits * span * span / 12)
    z = (x - mean) / (sd * math.sqrt(2.0))
    return 0.5 * (1.0 + np.vectorize(math.erf)(z))


def _loglik(lo: np.ndarray, hi: np.ndarray, hits: int, d: np.ndarray, tol: np.ndarray, least: np.ndarray,
            span: float) -> np.ndarray:
    """The log likelihood of each (lo[i], hi[i]) given exact reads `d` (give or take `tol`) and kills' `least`."""
    lo, hi = lo[:, None], hi[:, None]
    p = (1 - OUTLIER) * (_cdf(d[None, :] + tol[None, :], lo, hi, hits) - _cdf(d[None, :] - tol[None, :], lo, hi, hits))
    ll = np.log(p + OUTLIER * 2 * tol[None, :] / span).sum(axis=1)
    if len(least):
        q = (1 - OUTLIER_KILL) * (1.0 - _cdf(least[None, :], lo, hi, hits))
        ll = ll + np.log(q + OUTLIER_KILL * 0.5).sum(axis=1)
    return ll


def fit(reads: list[Read], hits: int) -> tuple[int, int] | None:
    """A hit's least and most damage (before the defence) that make `reads` likeliest, or None with under MIN_EXACT
    exact ones: on a grid (GRID) from 1 to the exact reads' median a hit for lo and from it to half again their most for
    hi, then a finer grid about the best. A kill's 'at least' more than KILL_SLACK over the most an exact read had isn't
    used: hers can't have done that (on the night's casts as follow.py credited them, kills of a first cast that a
    second made had a Horny Mushroom's 'at least' at its whole 403 against casts of 254-298, and widened the fit to
    76-264 a hit)."""
    exact = [r for r in reads if r.how == EXACT]
    if len(exact) < MIN_EXACT:
        return None
    d = np.array([r.value for r in exact], dtype=float)
    tol = np.array([max(TOL_MIN, r.tol) for r in exact], dtype=float)
    cap = float((d + tol).max()) * (1 + KILL_SLACK)
    least = np.array([r.value for r in reads if r.how == AT_LEAST and r.value <= cap], dtype=float)
    top = max(float(d.max()), float(least.max()) if len(least) else 0.0)
    span = 2.0 * top + 1.0  # (an outlier: anything from 0 to twice the most read, as likely each)
    hits = max(1, int(hits))
    med = max(1.0, float(np.median(d)) / hits)
    los, his = np.linspace(1.0, med, GRID[0]), np.linspace(med, 1.5 * top / hits + 1.0, GRID[1])
    lo, hi = (g.ravel() for g in np.meshgrid(los, his, indexing="ij"))
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    best = int(np.argmax(_loglik(lo, hi, hits, d, tol, least, span)))
    step_lo, step_hi = los[1] - los[0], his[1] - his[0]
    lo2, hi2 = (g.ravel() for g in np.meshgrid(np.linspace(lo[best] - step_lo, lo[best] + step_lo, 21),
                                                np.linspace(hi[best] - step_hi, hi[best] + step_hi, 21), indexing="ij"))
    keep = (lo2 >= 1.0) & (hi2 > lo2)
    lo2, hi2 = lo2[keep], hi2[keep]
    best = int(np.argmax(_loglik(lo2, hi2, hits, d, tol, least, span)))
    lo_, hi_ = int(round(lo2[best])), int(round(hi2[best]))
    return max(1, lo_), max(lo_ + 1, hi_)


class Book:
    """Her attacks' damage learned (`fit`), each attack's reads and fit kept in `path` (her own file: paths.of().damage)
    between runs: {attack key: {"hits": n, "reads": [Read.row, ...], "fit": {...}}}. `level`: her level now (None: not
    read), for the reads kept and used (LEVELS_BACK)."""

    def __init__(self, path: Path | None, log=print, level=None):
        self.path, self.log = path, log
        self.level = level or (lambda: None)
        self.data: dict[str, dict] = {}
        self._fits: dict[str, tuple[int, int] | None] = {}
        self._fit_at: dict[str, float] = {}
        self._new: dict[str, int] = {}
        self._dirty, self._saved_at = False, time.monotonic()
        self.version = 0  # one more at each fit that changed it (Attacks takes her attack again: Attacks.on_look)
        self._busy: threading.Thread | None = None
        self.threaded = THREADED
        if path is not None:
            try:
                self.data = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items()
                             if isinstance(v, dict)}
            except (OSError, ValueError):
                self.data = {}
            atexit.register(self._at_exit)
        for key, v in self.data.items():
            fitted = v.get("fit") or {}
            if isinstance(fitted.get("lo"), (int, float)) and isinstance(fitted.get("hi"), (int, float)):
                self._fits[key] = (int(fitted["lo"]), int(fitted["hi"]))

    def reads(self, key: str) -> list[Read]:
        """Attack `key`'s kept reads of her level and LEVELS_BACK under it (all, her level not read)."""
        rows = (self.data.get(key) or {}).get("reads") or []
        out = []
        for row in rows:
            try:
                out.append(Read.of(row))
            except (TypeError, ValueError):
                continue
        level = self.level()
        if level is not None:
            out = [r for r in out if r.level is None or r.level >= level - LEVELS_BACK]
        return out

    def add(self, key: str, hits: int, read: Read) -> None:
        """A read of attack `key` (`hits` hits a throw), kept (the newest KEEP), its fit made again within REFIT_S."""
        entry = self.data.setdefault(key, {"hits": hits, "reads": []})
        entry["hits"] = hits
        entry["reads"] = (entry.get("reads") or [])[-(KEEP - 1):] + [read.row()]
        self._new[key] = self._new.get(key, 0) + 1
        self._dirty = True
        now = time.monotonic()
        if read.how != MISS and now - self._fit_at.get(key, -1e18) >= REFIT_S and self._busy is None:
            self._fit_at[key] = now
            if self.threaded:
                self._busy = threading.Thread(target=self._refit, args=(key, hits), daemon=True, name="damage-fit")
                self._busy.start()
            else:
                self._refit(key, hits)
        if now - self._saved_at >= SAVE_S:
            self.save()

    def _refit(self, key: str, hits: int) -> None:
        """Attack `key`'s fit made again (`_fit`); one that fails keeps the last, said, never the hunt's end."""
        try:
            self._fit(key, hits)
        except Exception as e:  # noqa: BLE001
            self.log(f"    her {key}'s damage not fitted again: {type(e).__name__}: {e}")
        finally:
            self._busy = None

    def _fit(self, key: str, hits: int) -> None:
        reads = self.reads(key)
        fitted = fit(reads, hits)
        if fitted is None:
            return
        old = self._fits.get(key)
        self._fits[key] = fitted
        if fitted != old:
            self.version += 1
        n_exact = sum(r.how == EXACT for r in reads)
        n_least = sum(r.how == AT_LEAST for r in reads)
        misses, judged = self._miss_counts(key)
        self.data[key]["fit"] = {"lo": fitted[0], "hi": fitted[1], "exact": n_exact, "at_least": n_least,
                                 "misses": misses, "judged": judged, "level": self.level(),
                                 "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if old is None or abs(fitted[0] - old[0]) > 0.05 * max(old[0], 1) or abs(fitted[1] - old[1]) > 0.05 * old[1]:
            self.log(f"    her {key}'s damage as her throws showed it: {fitted[0]}-{fitted[1]} a hit before the "
                     f"defence, x{hits} a throw ({n_exact} bars, {n_least} kills"
                     + ("" if old is None else f"; was {old[0]}-{old[1]}") + ")")

    def _miss_counts(self, key: str) -> tuple[int, int]:
        """(misses, throws judged): its MISS reads, and those and its EXACT reads after a bar before (`barred`)."""
        reads = self.reads(key)
        misses = sum(r.how == MISS for r in reads)
        return misses, misses + sum(r.how == EXACT and r.barred for r in reads)

    def attack(self, prior: Attack | None) -> Attack | None:
        """`prior` with its lo and hi as learned (its key's fit, from MIN_EXACT exact reads on), else `prior`."""
        if prior is None:
            return None
        fitted = self._fits.get(prior.key)
        if fitted is None:
            return prior
        return replace(prior, lo=fitted[0], hi=fitted[1])

    def miss(self, key: str, default: float) -> float:
        """Her throws' share that took nothing (MISS reads among the throws judged), from MIN_MISSES judged on; else
        `default`."""
        misses, judged = self._miss_counts(key)
        return default if judged < MIN_MISSES else misses / judged

    def said(self, key: str) -> str:
        """Her `key`'s damage as learned, for a log line ("" with none)."""
        fitted = self._fits.get(key)
        if fitted is None:
            n = sum(r.how == EXACT for r in self.reads(key))
            return f"{key}: {n} of {MIN_EXACT} bars to learn its damage" if n else ""
        f = (self.data.get(key) or {}).get("fit") or {}
        return (f"{key}: {fitted[0]}-{fitted[1]} a hit as learned ({f.get('exact', '?')} bars, "
                f"{f.get('at_least', '?')} kills)")

    def _at_exit(self) -> None:
        """The last reads saved as the process ends (quietly: its log may be gone by then)."""
        self.log = lambda line: None
        self.save()

    def save(self, force: bool = False) -> None:
        """Her reads and fits written to `path` (a new file put in its place: never half written), if any changed."""
        if self.path is None or not (self._dirty or force) or not self.data:
            return
        self._saved_at = time.monotonic()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.data, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError as e:
            self.log(f"    her damage's file not saved: {e}")
