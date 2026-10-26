"""Her jumps' attacks in a run's telemetry (lab/telemetry): how often each kind of jump throw the stances chose fired
(the 'jump' records: behavior.fight.log_jump, one per jump throw or jump back carried out for what its attack
reaches; `kind` why: 'above', a monster on a platform above out of a star's reach from the ground, 'reach', one out
of it otherwise, 'passing', a jump forward along her way on a trip; `jump` the jump's kind: 'up' in place,
'forward', 'back'), and what came of each, as the looks after it saw it:

- 'kill': an EXP gain naming the target's kind came KILL_AFTER (0.1-0.85 s) after its attack key (the record's time
  and its key_ms), and no later attack of hers at that kind was keyed in between that it's that one's: a kill's gain
  shows 0.15-0.46 s after the key of the throw that made it, the server's, before her star even leaves (live on
  2026-10-25: D:/MapleHelper-scratch/analysis-1025/jumps-perception-skeptic/gain_delay.py). The gains are the
  perception service's own records when a *-perceive.jsonl in the same folder covers the run (the hunter's lag
  them: by more than 0.5 s for 38% of kills live), else the hunter's 'exp' records;
- 'hit': a look within HIT_S of its key saw a monster of its kind on its platform within NEAR of where it was with an
  HP bar lower than the last look before the key had it (or newly shown);
- 'nothing seen': neither (a miss, a star stopped, the hit hidden under the damage numbers, or a monster dead by then).

    python -m gamelab.analysis.jumps [FILES ...]     (default: the newest telemetry file)

Read only; prints a table by why and kind of jump, with the attack key's median ms after the jump key and how far
above her the targets were."""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..paths import TELEMETRY

# s after an attack key that an EXP gain naming its target's kind is its kill: the gain showed 0.15-0.46 s after the
# key live (median 0.29-0.33), up to 0.83 with the service's frames slow. Until 2026-10-26 any gain up to KILL_S 1.6 s
# after the jump record, the hunter's own: a ground throw's kill 1.4 s on was taken for the jump's (1 of the 6 jump
# records of 2026-10-25 23:43, jumps-perception-skeptic/outcome3.py).
KILL_AFTER = (0.1, 0.85)
HIT_S = 1.4    # a look this soon after the key that sees the target's bar lower: a hit
NEAR = 60.0    # map units: the target, seen again (it walks 20-60 units by then)


def _rows(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _bar(scene: dict, kind: str, platform: int, x: float) -> tuple[bool, float | None]:
    """Whether look `scene` sees a monster of `kind` on `platform` within NEAR of map x, and its bar (None: unhurt)."""
    near = [m for m in scene.get("monsters", ()) if m.get("kind") == kind and m.get("platform") == platform
            and abs(m.get("x", 1e9) - x) <= NEAR]
    if not near:
        return False, None
    m = min(near, key=lambda m: abs(m["x"] - x))
    return True, m.get("bar")


def keys_of(rows: list[dict]) -> list[tuple[float, str | None]]:
    """Every attack of hers in a run: when its key went down (an 'attack' record's time, plus the key_ms of the 'jump'
    record just before it, a jump's attack being that late in its plan) and the kind of its target (None: nothing)."""
    out, jump = [], None
    for r in rows:
        if r.get("topic") == "jump":
            jump = r
        elif r.get("topic") == "attack":
            late = jump is not None and 0 <= r["t"] - jump["t"] < 0.1
            out.append((r["t"] + (jump.get("key_ms", 0) / 1000 if late else 0.0), r.get("kind")))
    return out


def kills_of(gains: list[dict], keys: list[tuple[float, str | None]]) -> dict[float, float]:
    """Each gain naming a kind credited to the latest attack at that kind keyed KILL_AFTER before it: attack key time ->
    the gain's time."""
    out = {}
    for g in gains:
        for kind in g.get("kinds") or ():
            c = [t for t, k in keys if k == kind and KILL_AFTER[0] <= g["t"] - t <= KILL_AFTER[1]]
            if c:
                out[max(c)] = g["t"]
    return out


def outcomes(rows: list[dict], gains: list[dict] | None = None) -> list[dict]:
    """Each 'jump' record of a run with what came of it (see the module's docstring): its fields, `key` (when its attack
    key went down) and `outcome`. `gains`: the EXP gains to go by (the service's 'exp' records), by default the
    run's."""
    scenes = [r for r in rows if r.get("topic") == "scene" and r.get("her")]
    if gains is None:
        gains = [r for r in rows if r.get("topic") == "exp" and any(r.get("kinds") or ())]
    keys = keys_of(rows)
    jumps = []
    for r in rows:
        if r.get("topic") != "jump":
            continue
        key = r["t"] + r.get("key_ms", 0) / 1000
        own = min((k for k, _ in keys if -1e-6 <= k - key < 0.1), default=None)  # its attack record's key
        if own is None:  # (none: the jump's own)
            keys.append((key, r.get("target")))
            own = key
        jumps.append((r, key, own))
    killed = kills_of(gains, keys)
    out = []
    for r, key, own in jumps:
        kind, pid = r.get("target"), r.get("platform")
        tx = r.get("x", 0.0) + r.get("dx", 0.0)
        before = [s for s in scenes if s["t"] <= key]
        seen, was = _bar(before[-1], kind, pid, tx) if before else (False, None)
        outcome = "nothing seen"
        if own in killed:
            outcome = "kill"
        else:
            for s in scenes:
                if key < s["t"] <= key + HIT_S:
                    there, bar = _bar(s, kind, pid, tx)
                    if there and bar is not None and (was is None or bar < was - 1e-3):
                        outcome = "hit"
                        break
        out.append({**r, "key": key, "outcome": outcome})
    return out


def service_gains(path: Path, rows: list[dict]) -> list[dict] | None:
    """The perception service's EXP gains over a run's time (its 'exp' records with a gain), from a *-perceive.jsonl in
    the run's folder whose records cover it; None if there's none."""
    times = [r["t"] for r in rows if "t" in r]
    if not times:
        return None
    t0, t1 = min(times), max(times)
    for other in sorted(path.parent.glob("*-perceive.jsonl")):
        if other == path:
            continue
        gains, covers = [], False
        for r in _rows(other):
            if r.get("t", 0) < t0 - 5 or r.get("t", 0) > t1 + 5:
                continue
            covers = True
            if r.get("topic") == "exp" and r.get("gain"):
                gains.append(r)
        if covers:
            return gains
    return None


def table(jumps: list[dict], minutes: float | None = None) -> str:
    """The jumps by why and kind: how many (a minute, with `minutes`), their outcomes, the attack key's median ms after
    the jump key, the targets' median height above her."""
    by = defaultdict(list)
    for j in jumps:
        by[(j.get("kind"), j.get("jump"))].append(j)
    lines = [f"{'why':8s} {'jump':8s} {'n':>4s} {'/min':>5s}  {'kill':>4s} {'hit':>4s} {'none':>4s}  key ms  up"]
    for (why, jump), js in sorted(by.items(), key=lambda kv: -len(kv[1])):
        c = Counter(j["outcome"] for j in js)
        rate = f"{len(js) / minutes:5.2f}" if minutes else "    -"
        lines.append(f"{why or '-':8s} {jump or '-':8s} {len(js):4d} {rate}  {c['kill']:4d} {c['hit']:4d} "
                     f"{c['nothing seen']:4d}  {statistics.median(j.get('key_ms', 0) for j in js):6.0f}  "
                     f"{statistics.median(j.get('up', 0.0) for j in js):+4.0f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    files = [Path(a) for a in args] or sorted((p for p in TELEMETRY.glob("*.jsonl") if "-perceive" not in p.name),
                                              key=lambda p: p.stat().st_mtime)[-1:]
    for path in files:
        rows = _rows(path)
        times = [r["t"] for r in rows if "t" in r]
        minutes = (max(times) - min(times)) / 60 if len(times) > 1 else None
        gains = service_gains(path, rows)
        jumps = outcomes(rows, gains)
        print(f"{path.name}: {len(jumps)} jump throws chosen for their reach"
              + (f" in {minutes:.1f} min" if minutes else "")
              + ("; kills by the perception service's EXP gains" if gains is not None else ""))
        if jumps:
            print(table(jumps, minutes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
