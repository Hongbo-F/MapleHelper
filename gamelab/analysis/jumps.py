"""Her jumps' attacks in a run's telemetry (lab/telemetry): how often each kind of jump throw the stances chose fired
(the 'jump' records: behavior.fight.log_jump, one per jump throw or jump back carried out for what its attack
reaches; `kind` why: 'above', a monster on a platform above out of a star's reach from the ground, 'reach', one out
of it otherwise, 'passing', a jump forward along her way on a trip; `jump` the jump's kind: 'up' in place,
'forward', 'back'), and what came of each, as the looks after it saw it:

- 'kill': an EXP gain naming the target's kind came within KILL_S of the record (her star's damage shows about
  0.5-1.1 s after its key: STAR_DELAY_MS and its flight, and the gain a look later);
- 'hit': a look within HIT_S saw a monster of its kind on its platform within NEAR of where it was with an HP bar
  lower than the last look before the record had it (or newly shown);
- 'nothing seen': neither (a miss, a star stopped, or the hit hidden under the damage numbers).

    python -m gamelab.analysis.jumps [FILES ...]     (default: the newest telemetry file)

Read only; prints a table by why and kind of jump, with the attack key's median ms after the jump key and how far
above her the targets were."""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..paths import TELEMETRY

KILL_S = 1.6   # an EXP gain this soon after the record, naming the target's kind: its kill
HIT_S = 1.4    # a look this soon after it that sees the target's bar lower: a hit
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


def outcomes(rows: list[dict]) -> list[dict]:
    """Each 'jump' record of a run with what came of it (see the module's docstring): its fields and `outcome`."""
    scenes = [r for r in rows if r.get("topic") == "scene" and r.get("her")]
    exps = [r for r in rows if r.get("topic") == "exp"]
    out = []
    for r in rows:
        if r.get("topic") != "jump":
            continue
        t, kind, pid = r["t"], r.get("target"), r.get("platform")
        tx = r.get("x", 0.0) + r.get("dx", 0.0)
        before = [s for s in scenes if s["t"] <= t]
        seen, was = _bar(before[-1], kind, pid, tx) if before else (False, None)
        outcome = "nothing seen"
        if any(t < e["t"] <= t + KILL_S and kind in (e.get("kinds") or ()) for e in exps):
            outcome = "kill"
        else:
            for s in scenes:
                if t < s["t"] <= t + HIT_S:
                    there, bar = _bar(s, kind, pid, tx)
                    if there and bar is not None and (was is None or bar < was - 1e-3):
                        outcome = "hit"
                        break
        out.append({**r, "outcome": outcome})
    return out


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
    files = [Path(a) for a in args] or sorted(TELEMETRY.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)[-1:]
    for path in files:
        rows = _rows(path)
        times = [r["t"] for r in rows if "t" in r]
        minutes = (max(times) - min(times)) / 60 if len(times) > 1 else None
        jumps = outcomes(rows)
        print(f"{path.name}: {len(jumps)} jump throws chosen for their reach"
              + (f" in {minutes:.1f} min" if minutes else ""))
        if jumps:
            print(table(jumps, minutes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
