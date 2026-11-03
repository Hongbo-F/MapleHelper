"""The map's spawn pass, learned from the births perception sees (`SpawnClock`): when the server's next pass will fill
the free spawn points, so her positioning knows when a free point of a monster whose touch hurts is about to put one
on her (attack.danger's `respawn_risk`, behavior.hunt's spawn guard), and when it can't till the next.

The server fills the map at a pass every PERIOD_S, not monster by monster (sim.world's top, behavior.patrol's:
OrionAlpha's LifePool.tryCreateMob, measured on Tree Dungeon on 2026-10-29 and 30). A newborn fades in from its pass:
perception's spawn watch finds it at its spawn point with its alpha (perception.spawns, Mob.fading; the tracker's
'spawned' event, localization.tracking), so its birth was alpha / FADE_PER_S before the frame. Measured on both hours of
2026-11-02 (the perception service's 15 fps scenes, every fading sighting's start; D:/MapleHelper-scratch/cornered-fix/
respawn/live, clock_fit.py, clock_fit.txt): the passes every 7.1146 s on both (BugBaby's 22:46 hour, 3,204 sightings
over 504 periods, se 0.00002 s; RiceBaby's 00:17 hour, 1,418 over 379, se 0.00001), the same in every 10 minutes of
each; on RiceBaby's hour 95% of the starts within 0.1 s of the fitted passes (p1 -0.11, p99 +0.09 s), on BugBaby's 46%
within 0.1 and 76% within 0.3 (a share of that hour's scenes recorded up to 0.75 s after their frames: its scenes' own
times, not the frames'). Every sighting a newborn's: the watch had no false positive on its 153,474 labelled
point-frames. (The cornered report's 7.110 s was a count of the gaps between passes within 0.25 s of a multiple, on a
0.005 s grid: every period from 7.110 to about 7.12 counts all 357; the least was reported.) Replayed on each hour's
newborns as the tracker tells them (each one's first fading sighting: clock_replay.py, clock_replay.txt) it's sure 14 s
in and has the births within 0.055 s of its passes at p95. The simulator's own pass, sim.world.TICK_S, is 7.115:
0.0004 s apart, 0.02 s over a run of 5 minutes, so it isn't changed.

How (`see`, `birth`): each 'spawned' event's birth (one found fading: its alpha back along FADE_PER_S; one found solid
by a spawn point: its first sighting, SOLID_WEIGHT as much, as a re-found track by a point is one too); births within
SAME_PASS_S of each other one pass's; the passes of the last MEMORY_S kept. The phase: of the passes' times modulo the
period, the one with the most weight within INLIER_S of it (their weights falling by age, AGE_S), then a least squares
fit of the inliers to the grid (the period too once they span PERIOD_SPAN_S, kept within PERIOD_SLACK of PERIOD_S: the
server's period is a fixed one, PERIOD_S a sane default till then). Its spread (the inliers' residuals) and their weight
give the phase's standard error (`se`); `sure` once SURE_N passes' weight agree.

What it answers (`chance`): how likely an interval of hers overlaps a window about some pass ([pass + lo, pass + hi]),
the pass's time drawn from the fit (normal, its se); with no phase yet, as likely as the window's share of a period
(the phase not known: evenly). `next_pass`: the next pass's time after a moment. Pure: nothing here looks or presses a
key; times are the looks' and the feed's events' clock (Scene.t, tracking.Event.t)."""

import math
from collections import deque
from dataclasses import dataclass

from ..perception.spawns import FADE_PER_S

# s between the server's passes as measured live (above): the default, and the centre of what a fit may give.
PERIOD_S = 7.1146
PERIOD_SLACK = 0.05    # s a fitted period may be off PERIOD_S at most (a fit beyond it is taken for noise)
PERIOD_SPAN_S = 120.0  # s the inliers must span for the period to be fitted (else PERIOD_S, the phase alone)
SAME_PASS_S = 0.4      # s: births this close are one pass's (a pass's births live within 0.2 s of each other)
INLIER_S = 0.35        # s off the grid at most for a pass to count in the fit (live 95-97% within 0.1-0.2 s)
MEMORY_S = 600.0       # s of passes kept
AGE_S = 300.0          # s: a pass's weight in the phase's choice falls by exp(-age / this)
SOLID_WEIGHT = 0.3     # a birth found solid by a spawn point (no alpha) counts this much of one found fading in
SIGMA_MIN = 0.03       # s: the fit's residual spread no less than this (the frames' own step, 1/15 s, halved)
SURE_N = 1.5           # the passes' weight agreeing (age-weighed) for the clock to be sure: two births found fading


@dataclass
class Pass:
    """A pass seen: when (s, the births' weighted mean), how much it counts (its births' best weight)."""
    t: float
    w: float


class SpawnClock:
    """The map's spawn pass, as learned from the births seen (see the top)."""

    def __init__(self, period: float = PERIOD_S):
        self.default = period
        self.period = period
        self.phase: float | None = None   # a pass's time (s, the looks' clock), None till a birth is seen
        self.se = math.inf                # s: the phase's standard error
        self.weight = 0.0                 # the inliers' weight
        self.passes: deque[Pass] = deque()
        self._dirty = False
        self.births = 0                   # births taken, in all (the log's)

    # --- what it learns from -------------------------------------------------------------------------

    def see(self, events) -> int:
        """Take a look's events (Scene.events: tracking.Event; None: no feed): each 'spawned' one, a birth (`birth`);
        how many were taken."""
        n = 0
        for e in events or ():
            if getattr(e, "kind", None) != "spawned":
                continue
            a = (e.data or {}).get("fading")
            if a is not None:
                self.birth(e.t - float(a) / FADE_PER_S, 1.0)
            else:
                self.birth(e.t, SOLID_WEIGHT)
            n += 1
        return n

    def birth(self, t: float, w: float = 1.0) -> None:
        """A birth at time `t` (s) counting `w`: a pass's, or a new one."""
        self.births += 1
        for p in reversed(self.passes):
            if abs(p.t - t) <= SAME_PASS_S:
                if w > p.w:
                    p.t, p.w = (p.t * p.w + t * w) / (p.w + w), w
                self._dirty = True
                return
            if p.t < t - SAME_PASS_S:
                break
        self.passes.append(Pass(t, w))
        if len(self.passes) > 1 and self.passes[-1].t < self.passes[-2].t:  # (kept in time order)
            self.passes = deque(sorted(self.passes, key=lambda p: p.t))
        while self.passes and self.passes[0].t < self.passes[-1].t - MEMORY_S:
            self.passes.popleft()
        self._dirty = True

    def _fit(self) -> None:
        """The phase (and the period, once spanned) from the passes kept (see the top)."""
        self._dirty = False
        ps = list(self.passes)
        if not ps:
            self.phase, self.se, self.weight = None, math.inf, 0.0
            return
        latest = ps[-1].t
        P = self.period
        ws = [p.w * math.exp(-(latest - p.t) / AGE_S) for p in ps]
        best = None
        for c in ps:  # the candidate phase with the most weight within INLIER_S of it
            score = sum(w for p, w in zip(ps, ws) if _off(p.t, c.t, P) <= INLIER_S)
            if best is None or score > best[0]:
                best = (score, c.t)
        ref = best[1]
        ins = [(p, w) for p, w in zip(ps, ws) if _off(p.t, ref, P) <= INLIER_S]
        ks = [round((p.t - ref) / P) for p, _ in ins]
        W = sum(w for _, w in ins)
        span = (max(p.t for p, _ in ins) - min(p.t for p, _ in ins)) if ins else 0.0
        if span >= PERIOD_SPAN_S and len(set(ks)) >= 3:  # weighted least squares of t = phase + k P
            mk = sum(w * k for (_, w), k in zip(ins, ks)) / W
            mt = sum(w * p.t for p, w in ins) / W
            sxx = sum(w * (k - mk) ** 2 for (_, w), k in zip(ins, ks))
            sxy = sum(w * (k - mk) * (p.t - mt) for (p, w), k in zip(ins, ks))
            if sxx > 0:
                P = min(self.default + PERIOD_SLACK, max(self.default - PERIOD_SLACK, sxy / sxx))
            phase = mt - P * mk  # (the grid's pass at k 0: by `ref`)
        else:
            phase = ref + sum(w * _signed(p.t, ref, P) for p, w in ins) / W
        res = [_signed(p.t, phase, P) for p, _ in ins]
        spread = max(SIGMA_MIN, math.sqrt(sum(w * r * r for (_, w), r in zip(ins, res)) / W))
        self.period = P
        self.phase = phase + P * round((latest - phase) / P)  # (the pass nearest the latest seen)
        self.weight = W
        self.se = spread / math.sqrt(max(W, 1e-9))

    # --- what it answers -----------------------------------------------------------------------------

    def _now(self) -> None:
        if self._dirty:
            self._fit()

    @property
    def sure(self) -> bool:
        """Whether the passes seen agree enough to time the next ones (SURE_N of them)."""
        self._now()
        return self.phase is not None and self.weight >= SURE_N

    def next_pass(self, t: float) -> float | None:
        """The time of the first pass after `t` (None: no phase yet)."""
        self._now()
        if self.phase is None:
            return None
        return self.phase + self.period * math.floor((t - self.phase) / self.period + 1.0)

    def chance(self, t0: float, t1: float, lo: float, hi: float) -> float:
        """How likely [t0, t1] (s) overlaps [T + lo, T + hi] for a pass at T: some pass's T within [t0 - hi, t1 - lo],
        each pass's time normal about the fit's grid with its se (summed over the passes it may be, at most 1); with no
        phase yet the window's share of a period, the phase being anywhere."""
        self._now()
        a, b = t0 - hi, t1 - lo
        if b < a:
            return 0.0
        P = self.period
        if self.phase is None or not math.isfinite(self.se):
            return min(1.0, (b - a) / P)
        if b - a >= P:
            return 1.0
        se = max(self.se, 1e-6)
        k0 = math.floor((a - self.phase) / P - 3.0 * se / P - 1.0)
        k1 = math.ceil((b - self.phase) / P + 3.0 * se / P + 1.0)
        total = 0.0
        for k in range(k0, k1 + 1):
            T = self.phase + k * P
            total += _cdf((b - T) / se) - _cdf((a - T) / se)
        return max(0.0, min(1.0, total))

    def record(self) -> dict:
        """As telemetry data (the hunt's 'spawn_clock' record): its phase, period, se and weight."""
        self._now()
        return {"phase": None if self.phase is None else round(self.phase, 3), "period": round(self.period, 4),
                "se": None if not math.isfinite(self.se) else round(self.se, 3), "weight": round(self.weight, 2),
                "passes": len(self.passes), "births": self.births}


def _signed(t: float, ref: float, P: float) -> float:
    """t's offset from the grid through `ref` every P, in (-P/2, P/2]."""
    return (t - ref) - P * round((t - ref) / P)


def _off(t: float, ref: float, P: float) -> float:
    return abs(_signed(t, ref, P))


def _cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
