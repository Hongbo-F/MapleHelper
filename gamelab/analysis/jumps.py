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
hit (a permutation test within each base key, kind and room). `python -m gamelab jitter [FILES]` prints both. In the
simulator the outcome follows its own model of the pick (sim.world, as attack.reach's), so a pattern there is the
model's: the live grinds are where one means something.

What the analysis must allow for (the reviews of 2026-11-01): each attack's jitter is drawn from its own room (the
'jump' record's `window`), narrowed where the search found later keys unsure, so a large jitter comes only from attacks
with a wide room and the target's geometry would pass for the jitter's effect: the test shuffles within each room
(ROOM, with its base key, kind, turn and whether it's sure). The jitter is the press's where the player's answer has it
(`pressed_ms` less the base: the key as it went), and an attack pressed more than PRESS_OFF_MS off the key sent (her
actions turning her in the air at their own key where the stance had her facing it) isn't counted at all. The soonest's
jitter (0 to +15, sd 4.6 ms) is smaller than the game's own spread of a key (KEY_SLACK_MS: two of body.Lags' 7.7 ms sd,
seen at the game's 30 ms steps), so a pattern within it shows only over many attacks; the top's (-20 to +20) is wider.
Every jump attack is recorded from the review of 2026-11-01 on (behavior.fight.JITTER_ALL); before, only those whose
keys the search found sure."""

import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..attack import reach
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
        tx = (r.get("x") or 0.0) + (r.get("dx") or 0.0)  # (none: a jump attack with no monster that way, a jump back's)
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


def _keys_of_jumps(rows: list[dict]) -> dict[int, dict]:
    """Each 'jump' record's 'jump_keys' record (id of the jump record -> it): the first after it with its `n` (each
    hunter numbers its own from 1, and a telemetry file may hold several hunters': until the review of 2026-11-01
    joined by `n` alone, a later hunter's took an earlier one's)."""
    out, last = {}, {}
    for r in sorted((r for r in rows if r.get("topic") in ("jump", "jump_keys") and r.get("n") is not None),
                    key=lambda r: (r.get("t", 0.0), r.get("topic") != "jump")):
        if r["topic"] == "jump":
            last[r["n"]] = r
        elif (j := last.pop(r["n"], None)) is not None:
            out[id(j)] = r
    return out


def jitter_rows(rows: list[dict], gains: list[dict] | None = None, scenes: list[dict] | None = None) -> list[dict]:
    """Each 'jump' record of a run that has its jitter (`base_ms`, `jitter_ms`: behavior.fight.log_jump), with
    `thrown` and `pressed_ms` from its 'jump_keys' record (the first after it with its `n`: _keys_of_jumps; whether its
    keys went; the attack key's press after the jump key's as the player pressed them, None unknown), `pressed_jitter`
    (that less `base_ms`, else the jitter as sent), `otherwise` (pressed more than PRESS_OFF_MS off the key sent: not
    keyed as the record says), `outcome` and `source`: the simulator's own ('sim'), its first jump throw's or jump
    back's 'sim_attack' keyed within MATCH_S after the record, each taken once, if the run has them; else the looks'
    ('looks': outcomes). One whose keys didn't go is 'not thrown'."""
    keys_of = _keys_of_jumps(rows)
    sims = sorted((r["key"], r) for r in rows if r.get("topic") == "sim_attack"
                  and r.get("plan") in ("jump_throw", "jump_back"))
    looks = {} if sims else {(j["t"], j.get("n")): j["outcome"] for j in outcomes(rows, gains, scenes)}
    out, i = [], 0
    for r in sorted((r for r in rows if r.get("topic") == "jump"), key=lambda r: r["t"]):
        k = keys_of.get(id(r))
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
            row["otherwise"] = p is not None and abs(p - r["key_ms"]) > PRESS_OFF_MS
            out.append(row)
    return out


PRESS_OFF_MS = 5  # a press this much off the key sent wasn't keyed as its record says (her actions' turn in the air:
                  # 76-78 ms off; the player's own lateness 1-2 ms), and isn't counted (jitter_table, jitter_test)
BIN_MS = 5  # the jitter's bins: this wide about a multiple of it (attack.reach.KEY_CHECK_MS: the keys checked)


def _jitter(row: dict) -> float:
    """A jump attack's jitter as the player pressed it (`pressed_jitter`: the press less the base key, the jitter as
    sent where the press isn't known)."""
    return float(row.get("pressed_jitter", row["jitter_ms"]))


def _bin(row: dict) -> int:
    """The bin of a jump attack's jitter as pressed (_jitter): the nearest multiple of BIN_MS (its middle; +5 is +3 to
    +7), so the soonest's room (0 to +15) is 0, +5, +10, +15 and the top's (-20 to +20) -20 to +20, the end bins the
    narrower."""
    return BIN_MS * round(_jitter(row) / BIN_MS)


CLOSE = ("close", "tap first")  # the 'jump' records' kinds of a jump throw up close (attack.stance._from_here's): no shot


def _group(row: dict) -> tuple:
    """A jump attack's base key, the jump's kind, whether it turns her in the air, and whether it's a shot at what's out
    of a throw's reach ('shot': the stance's _jump_shot), a sure jump throw up close ('close': CLOSE), or one whose keys
    nothing checked ('unsure': the record's `sure` False, behavior.fight.JITTER_ALL; a jump back's among them)."""
    of = "unsure" if row.get("sure") is False else "close" if row.get("kind") in CLOSE else "shot"
    return (row.get("base_ms") or 0, row.get("jump") or "-", bool(row.get("turn")), of)


def _room(row: dict) -> tuple[int, int]:
    """The room its jitter was drawn from, from its base key (the record's `window`): (0, 15) the soonest's whole room,
    (-20, 20) the top's, narrower where the search found its later keys unsure."""
    lo, hi = row.get("window") or (row["key_ms"], row["key_ms"])
    base = row.get("base_ms") or 0
    return lo - base, hi - base


def _whole(row: dict) -> tuple[int, int]:
    """The whole room of its base key's jitter: the top's (attack.reach.TOP_KEY_MS) JITTER_TOP_MS either side, the
    soonest's JITTER_FIRST_MS later."""
    if row.get("base_ms") == reach.TOP_KEY_MS:
        return -reach.JITTER_TOP_MS, reach.JITTER_TOP_MS
    return 0, reach.JITTER_FIRST_MS


def _label(g: tuple) -> str:
    base, jump, turn, of = g[:4]
    return f"{base:4d} {jump:8s} {'yes' if turn else 'no':4s} {of:6s}"


_HEAD = f"{'base':>4s} {'jump':8s} {'turn':4s} {'of':6s}"


def _counted(rows: list[dict]) -> list[dict]:
    """The jump attacks thrown as their records say (not pressed otherwise: PRESS_OFF_MS)."""
    return [r for r in rows if r["thrown"] and not r.get("otherwise")]


def jitter_table(rows: list[dict]) -> str:
    """The jump attacks with jitter thrown as keyed (jitter_rows, _counted), by base key, jump kind, a turn in the air,
    shot, close or unsure (_group) and jitter bin (_bin): how many, kills, hits (a bar lower, the simulator's 'hit'),
    misses (the simulator's noise), wasted (WASTED), unmatched, the share that hit (HITS), the share of its attacks
    whose room was narrowed (`narrow`: not its whole room, _whole; a bin of those is the target's geometry as much as
    the jitter's), and the target's median distance and height."""
    thrown = _counted(rows)
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in thrown:
        by[(*_group(r), _bin(r))].append(r)
    lines = [f"{_HEAD} {'jitter':>6s} {'n':>4s} {'kill':>4s} {'hit':>4s} {'miss':>4s} {'none':>4s} {'?':>3s} "
             f"{'hit%':>5s} {'narrow':>6s}  {'dx':>4s} {'up':>4s}"]
    for g, rs in sorted(by.items(), key=lambda kv: kv[0]):
        c = Counter(r["outcome"] for r in rs)
        hit = sum(c[o] for o in HITS)
        narrow = sum(_room(r) != _whole(r) for r in rs)
        lines.append(f"{_label(g)} {f'{g[4]:+d}':>6s} "
                     f"{len(rs):4d} {c['kill']:4d} {c['hit']:4d} {c['miss']:4d} {sum(c[o] for o in WASTED):4d} "
                     f"{c['unmatched']:3d} {100 * hit / len(rs):5.0f} {100 * narrow / len(rs):5.0f}%  "
                     f"{statistics.median(abs(r.get('dx') or 0.0) for r in rs):4.0f} "
                     f"{statistics.median(r.get('up') or 0.0 for r in rs):+4.0f}")
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
    """Whether the jitter predicts a hit (HITS against the rest thrown as keyed, _counted; unmatched left out), the
    jitter as pressed (_jitter), within each base key, jump kind, turn, shot, close or unsure (_group) and room (_room:
    the hits shuffled among attacks drawn from the same room alone, so a room narrowed by the target's geometry passes
    for no jitter's effect; until the reviews of 2026-11-01 within each group whatever its rooms) with at least 2 hits
    and 2 not, and over all of them (stratified by those): how many, the hits' mean jitter against the others', and a
    permutation test's two-sided p (SHUFFLES, seeded by `seed`). A group's line pools its rooms' (a p of its own, its
    rooms stratified); the last line all of them. A small p (under 0.05) says the outcome goes with the jitter there; a
    large one, that this many attacks show no such pattern."""
    rng = random.Random(seed)
    by: dict[tuple, dict[tuple, list[tuple[float, bool]]]] = defaultdict(lambda: defaultdict(list))
    for r in _counted(rows):
        if r["outcome"] != "unmatched":
            by[_group(r)][_room(r)].append((_jitter(r), r["outcome"] in HITS))
    lines = [f"{_HEAD} {'n':>4s} {'hits':>4s} {'rooms':>5s} {'jitter, hits':>12s} {'others':>7s} {'p':>6s}"]
    tested = []
    for key, rooms in sorted(by.items(), key=lambda kv: kv[0]):
        strata = [g for g in rooms.values() if any(h for _, h in g) and not all(h for _, h in g)
                  and len({j for j, _ in g}) > 1]  # (a room with both outcomes and more than one jitter: it can tell)
        every = [x for g in rooms.values() for x in g]
        hits = [j for j, h in every if h]
        rest = [j for j, h in every if not h]
        told = [x for g in strata for x in g]
        if sum(h for _, h in told) < 2 or sum(not h for _, h in told) < 2:
            lines.append(f"{_label(key)} {len(every):4d} {len(hits):4d} {len(rooms):5d} {'':>12s} {'':>7s}      - "
                         f"(too few of one outcome within a room)")
            continue
        tested += strata
        _, pv = _perm_p(strata, shuffles, rng)
        lines.append(f"{_label(key)} {len(every):4d} {len(hits):4d} {len(rooms):5d} {statistics.fmean(hits):+12.1f} "
                     f"{statistics.fmean(rest):+7.1f} {pv:6.3f}")
    if tested:
        per, pv = _perm_p(tested, shuffles, rng)
        lines.append(f"all of those, within each room: the hits' jitter {per:+.1f} ms from their room's mean, "
                     f"p {pv:.3f} ({sum(len(g) for g in tested)} attacks, {len(tested)} rooms)")
    else:
        lines.append("nothing to test: no base key, kind and room with both outcomes")
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
        lines.append(f"{path.name}: {len(js)} jump attacks with jitter, {sum(r['thrown'] for r in js)} thrown, "
                     f"{sum(r.get('sure') is False for r in js)} of them not sure; "
                     + ("outcomes the simulator's own (its model's: a pattern there is the model's)" if sims else
                        "outcomes the looks' (" + ("the perception service's EXP gains and scenes" if gains is not None
                                                   else "her own") + ")"))
        every += js
    if not every:
        return lines + ["no jump attacks with jitter (recorded from 2026-11-01 on)"]
    sent = [r for r in every if r["thrown"] and r.get("pressed_ms") is not None]
    off = sum(bool(r.get("otherwise")) for r in every if r["thrown"])
    lines += ["", f"all: {len(every)} jump attacks, {sum(r['thrown'] for r in every)} thrown"
              + (f", {off} pressed more than {PRESS_OFF_MS} ms off the key sent (not counted)" if off else ""),
              jitter_table(every), "", "does the jitter predict a hit?", jitter_test(every)]
    if sent:
        off = [r["pressed_ms"] - r["key_ms"] for r in sent]
        lines += ["", f"the player's presses against the keys sent ({len(sent)}): median {statistics.median(off):+.1f} "
                      f"ms, at most {max(off, key=abs):+.1f}; {sum(abs(o) > 2 for o in off)} more than 2 ms off (her "
                      f"actions keying it otherwise: a turn in the air's key where the stance had her facing it)"]
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
