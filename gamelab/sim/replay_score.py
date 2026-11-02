"""The replay kit's scores (gamelab.sim.replay `score`): a dump of a recording (replay_run's) against that recording's
labels (tests/fixtures/recordings/<clip>/labels-<version>.json), class by class. Each `score_*` takes the dump's
frames (replay.load's, in order, with its head's t0 where times matter) and the label class's items, and gives plain
numbers; `score` all of them for one recording, `main` the command (a table, master's cached dumps beside if asked).

The classes and what each says (README, Replaying the recordings):
- onsets: the spawn watch's onsets. `of_record` the labelled ones the dump has at the same point and frame (master
  reproduces R2+R3's 197 of 197 and R1's 47 of 47); `false` the dump's onsets on no labelled birth (a point and frames
  where no newborn faded in: the user's zero-false rule, 0 to keep); `refinds` onsets of a birth after its first
  (P19); R1's `given`: the 30 newborns the state gave out as fading monsters, given out by the dump too.
- births: every birth in view (`found` by an onset at its point within FIND_FRAMES of its first visible frame), apart
  for those master's watch found and those it missed (P18: a fix's gain shows there); frames to the first onset and
  the alpha then (medians).
- touches: the dump's touches (worldstate's 'touch' events) against the labelled ones within TOUCH_S: found, by popup
  (P22: every labelled one showed one), how many frames late, its number right, and touches on no label (false).
- under_her: at each labelled frame (and the one either side) a monster of that kind in the state, or a track it
  carries, at her body (BODY_DX across, BODY_DY up or down: P15).
- kills: her gains' first and last bindings (the 'victim' events) against the labelled victim (within KILL_DX across
  and KILL_DY up or down), the first on the victim's level (KILL_DY: the platform credit, P14), first bindings right
  that were let go all the same; `strong` the labels decided on the frames (die1, the R1 lenses, by eye), `all` with
  master's unchecked final bindings too; unsure labels left out.
- casts (where the dump's feed lies beside it): her live casts at her dead, chosen after a labelled kill's gain showed
  (CAST_DEAD_S: its monster still drawn) at the victim's track in this dump (the one last at its place), and how many
  of them the dump's state at the frame shown before the cast was chosen had killed (the hunt leaves those out:
  behavior.ledger.Ledger.dead); and the casts at any other target it had killed.
- players: the state's players against the minimap's red dots every 15th frame (more or fewer than the dots), the ids
  a 5 minutes (P24's churn), and of the frames with a red dot by hers, those where the state has a player by her
  (P23)."""

import bisect
import json
import statistics
from collections import defaultdict
from pathlib import Path

FIND_FRAMES = 15   # an onset this many frames after a birth's first visible frame (3 before) finds it
REFIND_FRAMES = 30  # ... and onsets at its point up to this long after are its own (re-finds), not false
TOUCH_S = 0.35     # a touch told this near a labelled one's frame is it
BODY_DX, BODY_DY = 40.0, 40.0
KILL_DX, KILL_DY = 40.0, 25.0
CAST_DEAD_S = 1.6  # s after a kill's gain its monster is still drawn (last at 1.2 s at the median: tracking.REBIND_S)
CAST_DX = 45.0     # a cast's target is the track of its kind on its platform nearest its attack record's x, this near
STRONG = ("die1", "frames", "eye")
NEAR_PLAYER = (60.0, 60.0)  # a player this near her (across, up or down) is the one under her dot
SHORT_OF = {"horny-mushroom": "H", "green-mushroom": "G", "slime": "S"}  # (the dumps' kinds: replay_run's SHORT)


def _median(v):
    return None if not v else round(statistics.median(v), 3)


def onsets_of(frames: list[dict]) -> list[dict]:
    """The dump's onsets: its spawn watch's sightings ('sg') grouped by point and onset time, each at its first frame."""
    seen, out = set(), []
    for f in frames:
        for s in f.get("sg", ()):
            point, kind, x, y, a, since = s[0], s[1], s[2], s[3], s[4], s[8]
            if (point, since) not in seen:
                seen.add((point, since))
                out.append({"point": point, "kind": kind, "frame": f["i"], "x": x, "y": y, "a": a})
    return out


def score_onsets(frames: list[dict], items: list[dict], births: list[dict]) -> dict:
    mine = onsets_of(frames)
    at = {(o["point"], o["frame"]) for o in mine}
    of_record = sum(1 for o in items if (o["point"], o["frame"]) in at)
    false = [o for o in mine if not any(o["point"] == b["point"] and b["frame"] - 3 <= o["frame"]
                                        <= b["frame"] + REFIND_FRAMES for b in births)]
    out = {"labelled": len(items), "of_record": of_record, "onsets": len(mine), "false": len(false),
           "false_at": [[o["frame"], o["point"]] for o in false][:10]}
    per_birth = defaultdict(int)
    for o in mine:
        b = next((b for b in births if o["point"] == b["point"] and b["frame"] - 3 <= o["frame"]
                  <= b["frame"] + REFIND_FRAMES), None)
        if b is not None:
            per_birth[b["id"]] += 1
    out["refinds"] = sum(n - 1 for n in per_birth.values())
    given = [o for o in items if o.get("given")]
    if given:
        by_i = {f["i"]: f for f in frames}
        out["given"] = sum(1 for o in given if any(
            m[5] is not None and abs(m[1] - o["x"]) <= 20 and abs(m[2] - o["y"]) <= 20
            for i in range(o["frame"], o["frame"] + 4) for m in by_i.get(i, {}).get("m", ())))
        out["given_labelled"] = len(given)
    return out


def score_births(frames: list[dict], births: list[dict]) -> dict:
    mine = onsets_of(frames)
    out = {}
    for part, sel in (("seen_by_master", True), ("missed_by_master", False)):
        bs = [b for b in births if b["found"] is sel]
        found, late, alpha = 0, [], []
        for b in bs:
            hits = [o for o in mine if o["point"] == b["point"] and b["frame"] - 3 <= o["frame"] <= b["frame"]
                    + FIND_FRAMES]
            if hits:
                first = min(hits, key=lambda o: o["frame"])
                found += 1
                late.append(first["frame"] - b["frame"])
                alpha.append(first["a"])
        out[part] = {"births": len(bs), "found": found, "frames_late": _median(late), "alpha": _median(alpha)}
    out["under_her"] = sum(1 for b in births if b.get("under_her"))
    return out


def touches_of(frames: list[dict], t0: float) -> list[dict]:
    out = []
    for f in frames:
        for kind, track, et, d in f.get("ev", ()):
            if kind == "touch" and d.get("at") is not None:
                out.append({"t": d["at"] - t0, "how": d.get("how"), "number": d.get("number", d.get("text")),
                            "frame": f["i"]})
    return out


def score_touches(frames: list[dict], t0: float, items: list[dict]) -> dict:
    by_i = {f["i"]: f for f in frames}
    mine = touches_of(frames, t0)
    used, found, popup, late, number = set(), 0, 0, [], 0
    for lab in items:
        ft = by_i[lab["frame"]]["t"] if lab["frame"] in by_i else None
        if ft is None:
            continue
        cand = [(abs(m["t"] - ft), k) for k, m in enumerate(mine) if k not in used and abs(m["t"] - ft) <= TOUCH_S]
        if not cand:
            continue
        k = min(cand)[1]
        used.add(k)
        m = mine[k]
        found += 1
        popup += m["how"] == "popup"
        late.append(round((m["t"] - ft) * 15))
        number += lab.get("damage") is not None and str(m["number"]) == str(lab["damage"])
    return {"labelled": len(items), "found": found, "by_popup": popup, "frames_late": _median(late),
            "late_most": max(late, default=None), "number_right": number, "false": len(mine) - len(used),
            "told": len(mine)}


def score_under_her(frames: list[dict], items: list[dict]) -> dict:
    by_i = {f["i"]: f for f in frames}
    seen = []
    for lab in items:
        got = False
        for i in (lab["frame"] - 1, lab["frame"], lab["frame"] + 1):
            f = by_i.get(i)
            if not f or not f.get("her"):
                continue
            hx, hy = f["her"][0], f["her"][1]
            near = [m for m in f.get("m", ()) if m[0] == lab["kind"] and abs(m[1] - hx) <= BODY_DX
                    and abs(m[2] - hy) <= BODY_DY]
            near += [t for t in f.get("tr", ()) if t[2] == lab["kind"] and t[1] in ("live", "hidden", "dying")
                     and abs(t[3] - hx) <= BODY_DX and abs(t[4] - hy) <= BODY_DY]
            got = got or bool(near)
        seen.append(got)
    return {"labelled": len(items), "seen": sum(seen)}


def bindings_of(frames: list[dict], t0: float) -> dict:
    """Her gains' bindings in a dump: gain (s from t0, 3 decimals) -> [(frame, track, x, y)] in order, and the let-gos
    (gain -> tracks)."""
    by_i = {f["i"]: f for f in frames}
    binds, lets = defaultdict(list), defaultdict(set)

    def place(tid, at):
        for i in range(at, at - 8, -1):
            t = next((t for t in by_i.get(i, {}).get("tr", ()) if t[0] == tid), None)
            if t:
                return t[3], t[4]
        return None
    for f in frames:
        for kind, track, et, d in f.get("ev", ()):
            if kind == "victim" and not d.get("share"):
                p = place(track, f["i"])
                binds[round(d["gain"] - t0, 3)].append((f["i"], track, d["x"], None if p is None else p[1]))
            elif kind == "let_go":
                lets[round(d["gain"] - t0, 3)].add(track)
    return {"binds": binds, "lets": lets}


def score_kills(frames: list[dict], t0: float, items: list[dict]) -> dict:
    got = bindings_of(frames, t0)
    binds, lets = got["binds"], got["lets"]

    def right(b, victim):
        return b is not None and b[3] is not None and abs(b[2] - victim[0]) <= KILL_DX and \
            (victim[1] is None or abs(b[3] - victim[1]) <= KILL_DY)
    out = {}
    for part, ok in (("strong", lambda k: k["how"] in STRONG), ("all", lambda k: k["how"] != "unsure")):
        sel = [k for k in items if ok(k) and k.get("victim")]
        first = final = unbound = level = let_right = 0
        for k in sel:
            key = min(binds, key=lambda g: abs(g - k["gain"]), default=None)
            if key is None or abs(key - k["gain"]) > 0.03:
                unbound += 1
                continue
            b = binds[key]
            last = [x for x in b if x[1] not in lets.get(key, ())]
            first += right(b[0], k["victim"])
            final += right(last[-1] if last else None, k["victim"])
            level += b[0][3] is not None and k["victim"][1] is not None and abs(b[0][3] - k["victim"][1]) <= KILL_DY
            let_right += right(b[0], k["victim"]) and b[0][1] in lets.get(key, ())
        out[part] = {"labelled": len(sel), "first_right": first, "final_right": final, "unbound": unbound,
                     "first_level": level, "let_go_right": let_right}
    out["unsure"] = sum(1 for k in items if k["how"] == "unsure" or not k.get("victim"))
    out["let_go"] = sum(len(v) for v in lets.values())
    return out


def victim_track(by_i: dict, frame: int, kind: str, victim: list, span: tuple[int, int] = (-3, 26)) -> int | None:
    """The dump's track of `kind` last at a labelled victim's place (within KILL_DX and KILL_DY) about its gain's
    frame."""
    near: dict[int, float] = {}
    for i in range(frame + span[0], frame + span[1]):
        for t in by_i.get(i, {}).get("tr", ()):
            if t[2] == kind and abs(t[3] - victim[0]) <= KILL_DX and (victim[1] is None
                                                                      or abs(t[4] - victim[1]) <= KILL_DY):
                near[t[0]] = abs(t[3] - victim[0]) + (0.0 if victim[1] is None else 0.5 * abs(t[4] - victim[1]))
    return min(near, key=near.get, default=None)


def score_casts(frames: list[dict], t0: float, items: list[dict], throws: list[dict]) -> dict:
    """Her casts at her dead (see the top), from the feed's throws (each its attack record's kind, x, platform and when
    it was told)."""
    by_i = {f["i"]: f for f in frames}
    ts = [f["t"] for f in frames]
    victims = [(k, v) for k in items if k["how"] != "unsure" and k.get("victim")
               if (v := victim_track(by_i, k["frame"], k["kind"], k["victim"])) is not None]
    at_dead = left_out = other_left_out = 0
    for th in throws:
        if th.get("x") is None:
            continue
        told = th["told"] - t0
        j = bisect.bisect_right(ts, told - 0.02) - 1
        if j < 0:
            continue
        kind = SHORT_OF.get(th.get("kind"), th.get("kind"))
        cands = [t for t in frames[j].get("tr", ()) if t[2] == kind and abs(t[3] - th["x"]) <= CAST_DX
                 and (th.get("platform") is None or t[5] == th["platform"])]
        tgt = min(cands, key=lambda t: abs(t[3] - th["x"]), default=None)
        if tgt is None:
            continue
        killed = tgt[1] == "dead" or (tgt[1] == "dying" and tgt[9] is not None)
        if any(v == tgt[0] and 0.0 < told - k["gain"] <= CAST_DEAD_S for k, v in victims):
            at_dead += 1
            left_out += killed
        else:
            other_left_out += killed
    return {"at_her_dead": at_dead, "left_out": left_out, "still": at_dead - left_out,
            "other_left_out": other_left_out}


def score_players(frames: list[dict], items: list[dict]) -> dict:
    by_i = {f["i"]: f for f in frames}
    more = fewer = held = under = 0
    ids = set()
    for f in frames:
        ids.update(p[0] for p in f.get("pl", ()) if p[0] is not None)
    for i, reds, by_her in items:
        f = by_i.get(i)
        if f is None:
            continue
        n = len(f.get("pl", ()))
        more += n > reds
        fewer += n < reds
        if by_her:
            under += 1
            her = f.get("her")
            held += bool(her) and any(p[2] is not None and p[3] is not None and abs(p[2] - her[0]) <= NEAR_PLAYER[0]
                                      and abs(p[3] - her[1]) <= NEAR_PLAYER[1] for p in f.get("pl", ()))
    minutes = (frames[-1]["t"] - frames[0]["t"]) / 60 if len(frames) > 1 else 0
    return {"sampled": len(items), "more_than_dots": more, "fewer_than_dots": fewer,
            "ids_a_5_min": round(len(ids) / minutes * 5, 1) if minutes else None, "by_her_dot": under,
            "by_her_held": held}


def score(dump: Path, labels: dict) -> dict:
    """Every class's score of one dump against one recording's labels (labels-<version>.json's); her casts where the
    dump's feed lies beside it (<clip>.feed.json, as `run` and `cached` write it)."""
    from .replay import load
    head, frames, end = load(dump)
    t0 = head["t0"]
    cl = labels["classes"]
    births = cl["births"]["items"]
    feed = Path(dump).with_name(Path(dump).name.replace(".jsonl.gz", ".feed.json"))
    throws = json.loads(feed.read_text(encoding="utf-8"))["throws"] if feed.exists() else None
    return {"clip": labels["clip"], "labels": labels["version"], "frames": len(frames),
            "onsets": score_onsets(frames, cl["onsets"]["items"], births),
            "births": score_births(frames, births),
            "touches": score_touches(frames, t0, cl["touches"]["items"]),
            "under_her": score_under_her(frames, cl["under_her"]["items"]),
            "kills": score_kills(frames, t0, cl["kills"]["items"]),
            **({} if throws is None else {"casts": score_casts(frames, t0, cl["kills"]["items"], throws)}),
            "players": score_players(frames, cl["players"]["items"])}


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        elif not isinstance(v, list):
            out[f"{prefix}{k}"] = v
    return out


def total(scores: list[dict]) -> dict:
    """The recordings' scores added up (medians and other non-counts left out)."""
    sums: dict = {}
    for s in scores:
        for k, v in _flat({c: s[c] for c in ("onsets", "births", "touches", "under_her", "kills", "casts")
                           if c in s}).items():
            if isinstance(v, int) and not isinstance(v, bool) and not k.endswith(("late_most", "frames_late")):
                sums[k] = sums.get(k, 0) + v
    return sums


def main(a, clips) -> int:
    """`replay score`: the dumps in --dumps (a run's folder), and --base's cached ones beside if asked (never made
    here: `cached` makes them, a job for the CPU rules' wrapper); a table per recording and the total, against the
    labels of --labels' version, as JSON too (--json)."""
    from .replay import LABELS_VERSION, cached
    version = getattr(a, "labels", None) or LABELS_VERSION
    sides: dict[str, dict[str, Path]] = {}
    if a.dumps:
        sides["dumps"] = {c.folder: a.dumps / f"{c.folder}.jsonl.gz" for c in clips
                          if (a.dumps / f"{c.folder}.jsonl.gz").exists()}
    if a.base or not a.dumps:
        base = a.base or "master"
        sides[base] = cached(clips, base, 0, 0, 1, make=False)
        if len(sides[base]) < len(clips):
            print(f"replay: {base}'s dumps aren't all kept; to make them (paced, about 5 min, through the CPU rules' "
                  f"wrapper): python -m gamelab.sim.replay cached --commit {base}")
    result: dict = {}
    for side, dumps in sides.items():
        result[side] = {}
        for c in clips:
            if c.folder not in dumps or not c.labels_of(version).exists():
                continue
            labels = json.loads(c.labels_of(version).read_text(encoding="utf-8"))
            result[side][c.name] = score(dumps[c.folder], labels)
        result[side]["total"] = total([v for k, v in result[side].items() if k != "total"])
    names = list(result)
    for clip in [c.name for c in clips] + ["total"]:
        rows = {side: (_flat(result[side][clip]) if clip in result[side] else {}) for side in names}
        keys = [k for k in dict.fromkeys(k for r in rows.values() for k in r) if k not in ("clip", "labels")]
        if not keys:
            continue
        print(f"\n{clip}" + "".join(f"  {s:>14}" for s in names))
        for k in keys:
            print(f"  {k:34}" + "".join(f"  {str(rows[s].get(k, '')):>14}" for s in names))
    if a.json:
        a.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0
