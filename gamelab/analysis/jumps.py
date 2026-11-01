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
  HP bar lower than the last look before the key had it (or newly shown): the perception service's own scenes when its
  records cover the run (10 a second; the hunter's looks between her keys missed the 23:48:51 'above' jump's hit, its
  target's bar 0.46 0.8 s after the key in the service's), else the hunter's looks;
- 'nothing seen': neither (a miss, a star stopped, the hit hidden under the damage numbers, or a monster dead by then).

    python -m gamelab.analysis.jumps [FILES ...]     (default: the newest telemetry file)

Read only; prints a table by why and kind of jump, with the attack key's median ms after the jump key and how far
above her the targets were.

Her jump attacks' jitter (from 2026-11-01: two keys a jump, the soonest and the top's, each jittered a little as it's
carried out, behavior.fight.jittered; the user: "add a bit jitter to it so we still have some variation and can later
on analyze if the jitter can show any pattern"): `jitter_rows` joins each 'jump' record with jitter (`base_ms`,
`jitter_ms`) to its 'jump_keys' record (whether its keys went, the attack key as the player pressed it) and its outcome:
in a simulated run's telemetry the simulator's own ('sim_attack' records: sim.scenarios.Sim), else the looks' as above;
`jitter_table` tabulates them by base key, jump kind and jitter bin, `jitter_test` asks whether the jitter predicts a
hit (a permutation test within each base key and kind). `python -m gamelab jitter [FILES]` prints both. In the
simulator the outcome follows its own model of the pick (sim.world, as attack.reach's), so a pattern there is the
model's: the live grinds are where one means something."""

import json
import random
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


def outcomes(rows: list[dict], gains: list[dict] | None = None, scenes: list[dict] | None = None) -> list[dict]:
    """Each 'jump' record of a run with what came of it (see the module's docstring): its fields, `key` (when its attack
    key went down) and `outcome`. `gains`: the EXP gains to go by (the service's 'exp' records), by default the
    run's; `scenes`, the looks to see hits by (the service's 'scene' records), by default the run's."""
    if scenes is None:
        scenes = [r for r in rows if r.get("topic") == "scene" and r.get("her")]
    scenes = [r for r in scenes if r.get("monsters") is not None]
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
    got = service_records(path, rows)
    return None if got is None else got[0]


def service_records(path: Path, rows: list[dict]) -> tuple[list[dict], list[dict]] | None:
    """The perception service's EXP gains (its 'exp' records with a gain) and scenes placing her over a run's time, from
    a *-perceive.jsonl in the run's folder whose records cover it; None if there's none."""
    times = [r["t"] for r in rows if "t" in r]
    if not times:
        return None
    t0, t1 = min(times), max(times)
    for other in sorted(path.parent.glob("*-perceive.jsonl")):
        if other == path:
            continue
        gains, scenes, covers = [], [], False
        for r in _rows(other):
            if r.get("t", 0) < t0 - 5 or r.get("t", 0) > t1 + 5:
                continue
            covers = True
            if r.get("topic") == "exp" and r.get("gain"):
                gains.append(r)
            elif r.get("topic") == "scene" and r.get("her"):
                scenes.append(r)
        if covers:
            return gains, scenes
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


# --- her jump attacks' jitter (from 2026-11-01) -------------------------------------------------------------------
MATCH_S = 1.5  # s after a 'jump' record within which the simulator's first jump attack keyed is its own (as
               # sim.harness.CHOSEN_KEY_S: her lock waited out first, the attack at most 0.3 s into the jump)
HITS = ("kill", "hit")  # outcomes that hit (the looks': a kill credited, a bar lower; the simulator's: its own)
WASTED = ("stopped", "nothing", "nothing seen")  # ... and that hit nothing (the simulator's: a star stopped, none
                                                 # picked; the looks': neither a kill nor a hit seen)
SHUFFLES = 2000  # the permutation test's shuffles (seeded: the same answer when run again)


def jitter_rows(rows: list[dict], gains: list[dict] | None = None, scenes: list[dict] | None = None) -> list[dict]:
    """Each 'jump' record of a run that has its jitter (`base_ms`, `jitter_ms`: behavior.fight.log_jump), with
    `thrown` and `pressed_ms` from its 'jump_keys' record (by `n`: whether its keys went; the attack key's press after
    the jump key's as the player pressed them, None unknown), `pressed_jitter` (that less `base_ms`, else the jitter as
    sent), `outcome` and `source`: the simulator's own ('sim'), its first jump throw's or jump back's 'sim_attack' keyed
    within MATCH_S after the record, each taken once, if the run has them; else the looks' ('looks': outcomes). One
    whose keys didn't go is 'not thrown'."""
    keys = {r.get("n"): r for r in rows if r.get("topic") == "jump_keys"}
    sims = sorted((r["key"], r) for r in rows if r.get("topic") == "sim_attack"
                  and r.get("plan") in ("jump_throw", "jump_back"))
    looks = {} if sims else {(j["t"], j.get("n")): j["outcome"] for j in outcomes(rows, gains, scenes)}
    out, i = [], 0
    for r in sorted((r for r in rows if r.get("topic") == "jump"), key=lambda r: r["t"]):
        k = keys.get(r.get("n")) if r.get("n") is not None else None
        thrown = True if k is None else bool(k.get("thrown"))
        row = {**r, "thrown": thrown, "pressed_ms": None if k is None else k.get("pressed_ms")}
        if sims:
            outcome, source = "unmatched", "sim"
            if thrown:
                while i < len(sims) and sims[i][0] < r["t"] - 1e-6:
                    i += 1
                if i < len(sims) and sims[i][0] <= r["t"] + MATCH_S:
                    outcome = sims[i][1].get("outcome")
                    row["took"] = sims[i][1].get("kind")
                    i += 1
        else:
            outcome, source = looks.get((r["t"], r.get("n")), "nothing seen"), "looks"
        if not thrown:
            outcome = "not thrown"
        row.update(outcome=outcome, source=source)
        if "jitter_ms" in r:
            p = row["pressed_ms"]
            row["pressed_jitter"] = (p - r["base_ms"]) if p is not None and r.get("base_ms") is not None \
                else r["jitter_ms"]
            out.append(row)
    return out


BIN_MS = 5  # the jitter's bins: this wide about a multiple of it (attack.reach.KEY_CHECK_MS: the keys checked)


def _bin(row: dict) -> int:
    """The bin of a jump attack's jitter as sent: the nearest multiple of BIN_MS (its middle; +5 is +3 to +7), so the
    soonest's room (0 to +15) is 0, +5, +10, +15 and the top's (-20 to +20) -20 to +20, the end bins the narrower."""
    return BIN_MS * round(row["jitter_ms"] / BIN_MS)


def _group(row: dict) -> tuple:
    """A jump attack's base key, the jump's kind and whether it turns her in the air."""
    return row.get("base_ms"), row.get("jump"), bool(row.get("turn"))


def jitter_table(rows: list[dict]) -> str:
    """The jump attacks with jitter thrown (jitter_rows), by base key, jump kind (and a turn in the air) and jitter bin
    (_bin): how many, kills, hits (a bar lower, the simulator's 'hit'), misses (the simulator's noise), wasted
    (WASTED), unmatched, the share that hit (HITS), and the target's median distance and height."""
    thrown = [r for r in rows if r["thrown"]]
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in thrown:
        by[(*_group(r), _bin(r))].append(r)
    lines = [f"{'base':>4s} {'jump':8s} {'turn':4s} {'jitter':>9s} {'n':>4s} {'kill':>4s} {'hit':>4s} {'miss':>4s} "
             f"{'none':>4s} {'?':>3s} {'hit%':>5s}  {'dx':>4s} {'up':>4s}"]
    for (base, jump, turn, b), rs in sorted(by.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or "", kv[0][2],
                                                                     kv[0][3])):
        c = Counter(r["outcome"] for r in rs)
        hit = sum(c[o] for o in HITS)
        lines.append(f"{base or 0:4d} {jump or '-':8s} {'yes' if turn else 'no':4s} {f'{b:+d}':>9s} "
                     f"{len(rs):4d} {c['kill']:4d} {c['hit']:4d} {c['miss']:4d} {sum(c[o] for o in WASTED):4d} "
                     f"{c['unmatched']:3d} {100 * hit / len(rs):5.0f}  "
                     f"{statistics.median(abs(r.get('dx', 0.0)) for r in rs):4.0f} "
                     f"{statistics.median(r.get('up', 0.0) for r in rs):+4.0f}")
    return "\n".join(lines)


def _perm_p(groups: list[list[tuple[float, bool]]], shuffles: int, rng: random.Random) -> tuple[float, float]:
    """A stratified permutation test of whether the jitter predicts a hit: the statistic the hits' jitter less their
    group's mean jitter, summed over the groups (`groups`: (jitter, hit) each), the hits shuffled within each group;
    returns (the statistic per hit, two-sided p)."""
    def stat(gs) -> float:
        return sum(sum(j for j, h in g if h) - sum(h for _, h in g) * statistics.fmean(j for j, _ in g) for g in gs)
    t0 = stat(groups)
    hits = sum(h for g in groups for _, h in g)
    over = 0
    for _ in range(shuffles):
        shuffled = []
        for g in groups:
            hs = [h for _, h in g]
            rng.shuffle(hs)
            shuffled.append([(j, h) for (j, _), h in zip(g, hs)])
        over += abs(stat(shuffled)) >= abs(t0) - 1e-9
    return (t0 / hits if hits else 0.0), (over + 1) / (shuffles + 1)


def jitter_test(rows: list[dict], shuffles: int = SHUFFLES, seed: int = 0) -> str:
    """Whether the jitter predicts a hit (HITS against the rest thrown, unmatched left out), within each base key and
    jump kind with at least 2 hits and 2 not, and over all of them (stratified by those): how many, the hits' mean
    jitter against the others', and a permutation test's two-sided p (SHUFFLES, seeded by `seed`). A small p (under
    0.05) says the outcome goes with the jitter there; a large one, that this many attacks show no such pattern."""
    rng = random.Random(seed)
    by: dict[tuple, list[tuple[float, bool]]] = defaultdict(list)
    for r in rows:
        if r["thrown"] and r["outcome"] != "unmatched":
            by[_group(r)].append((float(r["jitter_ms"]), r["outcome"] in HITS))
    lines = [f"{'base':>4s} {'jump':8s} {'turn':4s} {'n':>4s} {'hits':>4s} {'jitter, hits':>12s} {'others':>7s} "
             f"{'p':>6s}"]
    tested = []
    for (base, jump, turn), g in sorted(by.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or "", kv[0][2])):
        hits = [j for j, h in g if h]
        rest = [j for j, h in g if not h]
        if len(hits) < 2 or len(rest) < 2:
            lines.append(f"{base or 0:4d} {jump or '-':8s} {'yes' if turn else 'no':4s} {len(g):4d} {len(hits):4d} "
                         f"{'':>12s} {'':>7s}      - (too few of one outcome)")
            continue
        tested.append(g)
        _, pv = _perm_p([g], shuffles, rng)
        lines.append(f"{base or 0:4d} {jump or '-':8s} {'yes' if turn else 'no':4s} {len(g):4d} {len(hits):4d} "
                     f"{statistics.fmean(hits):+12.1f} {statistics.fmean(rest):+7.1f} {pv:6.3f}")
    if tested:
        per, pv = _perm_p(tested, shuffles, rng)
        lines.append(f"all of those, within each: the hits' jitter {per:+.1f} ms from their group's mean, p {pv:.3f} "
                     f"({sum(len(g) for g in tested)} attacks)")
    else:
        lines.append("nothing to test: no base key and kind with 2 hits and 2 not")
    return "\n".join(lines)


def jitter_report(paths: list[Path]) -> list[str]:
    """The jitter report over telemetry files (`python -m gamelab jitter`): each file's jump attacks with jitter and
    where their outcomes are from, then all of them together, by jitter bin (jitter_table) and the test (jitter_test),
    and the player's presses against the keys sent."""
    lines, every = [], []
    for path in paths:
        rows = _rows(path)
        sims = any(r.get("topic") == "sim_attack" for r in rows)
        gains = scenes = None
        if not sims and (got := service_records(path, rows)) is not None:
            gains, scenes = got
        js = jitter_rows(rows, gains, scenes)
        if not js:
            continue
        lines.append(f"{path.name}: {len(js)} jump attacks with jitter, {sum(r['thrown'] for r in js)} thrown; "
                     + ("outcomes the simulator's own (its model's: a pattern there is the model's)" if sims else
                        "outcomes the looks' (" + ("the perception service's EXP gains and scenes" if gains is not None
                                                   else "her own") + ")"))
        every += js
    if not every:
        return lines + ["no jump attacks with jitter (recorded from 2026-11-01 on)"]
    sent = [r for r in every if r["thrown"] and r.get("pressed_ms") is not None]
    lines += ["", f"all: {len(every)} jump attacks, {sum(r['thrown'] for r in every)} thrown", jitter_table(every), "",
              "does the jitter predict a hit?", jitter_test(every)]
    if sent:
        off = [r["pressed_ms"] - r["key_ms"] for r in sent]
        lines += ["", f"the player's presses against the keys sent ({len(sent)}): median {statistics.median(off):+.1f} "
                      f"ms, at most {max(off, key=abs):+.1f}"]
    return lines


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    files = [Path(a) for a in args] or sorted((p for p in TELEMETRY.glob("*.jsonl") if "-perceive" not in p.name),
                                              key=lambda p: p.stat().st_mtime)[-1:]
    for path in files:
        rows = _rows(path)
        times = [r["t"] for r in rows if "t" in r]
        minutes = (max(times) - min(times)) / 60 if len(times) > 1 else None
        got = service_records(path, rows)
        gains, scenes = got if got is not None else (None, None)
        jumps = outcomes(rows, gains, scenes)
        print(f"{path.name}: {len(jumps)} jump throws chosen for their reach"
              + (f" in {minutes:.1f} min" if minutes else "")
              + ("; kills and hits by the perception service's EXP gains and scenes" if gains is not None else ""))
        if jumps:
            print(table(jumps, minutes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
