"""One recording through one code tree's perception frame step, as the perception service runs it with a hunt attached:
the child process of `python -m gamelab.sim.replay run` (gamelab.sim.replay, which says what it's for and
how to read what it writes). It is run as a script by its path, never imported, so that the tree it runs (`--root`: a
checkout, or a commit's export) is the `gamelab` it imports: this file's own folder is taken off sys.path first, and
nothing of the tree is imported before it's put first on it.

    python replay_run.py --root TREE --frames FOLDER --feed FEED.json --out DUMP.jsonl.gz [--side SIDE.json]
                         [--start I] [--limit N] [--unpaced] [--data GAMEDATA_DATA]

What it builds is what `service.screen_world` builds for the live service, with saved frames for the screen
(analysis.bench.SavedEyes: the same Eyes, its captures handed over): the map's monster detector (sprites.detector_for)
for the game's own 1366x768 (screen.NATIVE), her name tag's template from her calibration (analysis.calibration.load),
the map's minimap picture, Senses with her Shape (character.active: GAMELAB_CHARACTER, set by the parent) and the
look band (attack.reach.LOOK_UP, LOOK_DOWN, as `perceive` has it), the world step (worldstate.WorldStep) on the
service's schedule; then as `Service._tune` sets it while a command that hunts and tells her throws is attached (the
tour was, through every frame of R1-R3: their perceive.log): the hunt's schedule (service.HUNTING) and her throws
known (tracker.throws_known). OpenCV on one thread and Windows' timers at 1 ms (service.settle).

Each frame is handed over at its own time in the recording (paced: the real clock waits for it, 15 a second as the
service looked), the next one read from disk in the slack after the pass, never during one; `--unpaced` runs them back
to back (the same dump where nothing in the step reads the real clock: the parent's `check` says whether it is). The
step's clock is the live service's (time.monotonic's of that night): the feed's `t0` is frame 0's time on it, so every
time the step makes (her gains, her touches' popups, the tracks' times) reads against the live telemetry's own. What
the service took from its clients goes in from the feed (gamelab.sim.replay.feed: the hunt's records): her throws
(`step.throw`, each before the first frame shown after it was told, as the service's poll took it before its next
frame) and the drops the hunt asked for (an `Asks` on the frame it asked of). A frame missing from the recording (a
gap: the game not in front) is told as not visible, as feed.run does.

The dump (gzip, mtime 0, so a dump made twice is the same to the byte where the step is): a head line, then one line a
frame, then an end line with the step's, tracker's and spawn watch's counts. A frame's line (map units, rounded):
  i, t          the frame's index and time in the recording (its index.jsonl's)
  her           [x, y, platform, how, mode] (None: not placed); v [view x, y], vs (view registered), bad (why, 60 chars)
  hp, mp        her bars' shares; hpn her HP number [cur, max]; lv her level
  m             the state's monsters: [kind (H/G/S...), x, y, platform, track, fading, facing, bar, box x, y, w, h
                (look px), score, frame]
  tr            the tracks: [id, state, kind, x, y, platform, fading, how, bar, gain, cause]
  ev            the events: [kind, track, t - t0, data (its plain values)]
  pl            the other players: [id, kind, x, y, platform]
  sg            the spawn watch's sightings this frame before the hand-over: [point, kind, x, y, a, r2, res, onset,
                since - t0]
  wd            the whole screen's detections: [kind, foot x, feet y (look px), score, frame]
  loot          the drops asked for, where asked: [[x, y, platform, mesos]]
  say           what the eyes, senses or step logged on this frame, if anything
The pass's times (the real clock's) go in the side file instead (`--side`, JSON: per frame [i, late ms, pass ms, look
ms, whole screen ms, spawn watch ms], the lab files the run opened, the wall time it took), never in the dump: timings
are compared only between runs made side by side (`replay bench`), never against a cached one."""

import argparse
import gzip
import hashlib
import inspect
import io
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

FORMAT = 1
MAP_ID = 10002073
SHORT = {"horny-mushroom": "H", "green-mushroom": "G", "slime": "S"}


def _parse(argv):
    ap = argparse.ArgumentParser(prog="replay_run.py")
    ap.add_argument("--root", required=True, help="the code tree to run: a checkout or a commit's export")
    ap.add_argument("--frames", required=True, help="the recording's frames (index.jsonl and its PNGs)")
    ap.add_argument("--feed", required=True, help="the feed (JSON): t0, throws, asks (gamelab.sim.replay.feed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", default="", help="the side file: the pass's times and what the run read")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--unpaced", action="store_true", help="the frames back to back, not at their own times")
    ap.add_argument("--data", default="", help="gamedata's data folder, where the tree has none of its own")
    return ap.parse_args(argv)


def _r1(v):
    return None if v is None else round(float(v), 1)


def _r3(v):
    return None if v is None else round(float(v), 3)


def _plain(data: dict) -> dict:
    """An event's data as plain JSON values (what the pipe would carry of it): numbers, strings, None, and lists of
    them; anything else as its str."""
    out = {}
    for k, v in data.items():
        if v is None or isinstance(v, (bool, int, str)):
            out[k] = v
        elif isinstance(v, float):
            out[k] = round(v, 4)
        elif isinstance(v, (list, tuple)) and all(x is None or isinstance(x, (bool, int, float, str)) for x in v):
            out[k] = [round(x, 4) if isinstance(x, float) else x for x in v]
        else:
            out[k] = str(v)
    return out


class _Dump:
    """A gzip JSON-lines file written whole: to a .part beside it, renamed into place at the end; gzip's header without
    a time or a name, so the same lines give the same bytes."""

    def __init__(self, path: Path):
        self.path, self.part = path, path.with_name(path.name + ".part")
        self._raw = open(self.part, "wb")
        self._gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._text = io.TextIOWrapper(self._gz, encoding="utf-8", newline="\n")

    def write(self, rec: dict) -> None:
        self._text.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._text.close()
        self._raw.close()
        os.replace(self.part, self.path)


def main(argv=None) -> int:
    a = _parse(argv)
    root = Path(a.root).resolve()
    here = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or ".")) != here]
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    lab = os.path.normcase(str(root / "gamelab" / "lab")) + os.sep
    opened: set[str] = set()

    def audit(event, args):  # (the lab files the run opens: the parent keys a cached dump by them)
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            full = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
            if full.startswith(lab):
                opened.add(full[len(lab):].replace(os.sep, "/"))
    sys.addaudithook(audit)

    import gamedata.paths as gpaths  # (first: the modules after it take its paths as they import)
    if a.data:
        d = Path(a.data)
        gpaths.DATA, gpaths.COPY, gpaths.OUT, gpaths.TABLES = d, d / "copy", d / "out", d / "out" / "tables"
    import ctypes

    import cv2

    from gamelab import character
    from gamelab.analysis import calibration
    from gamelab.analysis.bench import SavedEyes
    from gamelab.attack import reach
    from gamelab.localization import feed, mapdata, service, worldstate
    from gamelab.localization.senses import Senses
    from gamelab.perception.frames import Frame
    from gamelab.perception.sprites import detector_for
    from gamelab.screen import NATIVE

    if not Path(feed.__file__).resolve().is_relative_to(root):
        raise SystemExit(f"replay_run: gamelab came from {feed.__file__}, not the tree {root}")
    cv2.setNumThreads(1)
    if sys.platform == "win32":
        ctypes.WinDLL("winmm").timeBeginPeriod(1)

    the_feed = json.loads(Path(a.feed).read_text(encoding="utf-8"))
    frames = Path(a.frames)
    rows = [json.loads(line) for line in open(frames / "index.jsonl", encoding="utf-8")][a.start:]
    if a.limit:
        rows = rows[:a.limit]
    said: list[str] = []
    model = mapdata.load(MAP_ID)
    template = calibration.load(NATIVE).template
    eyes = SavedEyes(template, detector_for(model, NATIVE), NATIVE, mapdata.load_minimap_picture(model.id))
    eyes.log = said.append
    senses = Senses(eyes, model, (reach.LOOK_UP, reach.LOOK_DOWN), shape=character.active().shape)
    if hasattr(senses, "log"):
        senses.log = said.append
    if senses.locator is None:
        raise SystemExit("replay_run: no locator: her models aren't in the tree's lab")
    step = worldstate.WorldStep(senses, False, service.SCHEDULE)
    step.schedule = getattr(service, "HUNTING", service.SCHEDULE)  # (Service._tune with a hunt attached)
    step.tracker.throws_known = True                                # (... that tells her throws)
    runner = feed.Runner(None, eyes, step)
    watch = getattr(step, "spawns", None)

    sights: list = []
    wide: list = []
    if watch is not None:
        looked = watch.look

        def look(*args, **kw):
            got = looked(*args, **kw)
            sights[:] = got
            return got
        watch.look = look
    if hasattr(step, "_wide"):
        searched = step._wide

        def search(*args, **kw):
            got = searched(*args, **kw)
            wide[:] = got
            return got
        step._wide = search

    t0 = float(the_feed["t0"])
    first_t = t0 + rows[0]["t"] if rows else t0
    throws = deque(th for th in sorted(the_feed.get("throws", []), key=lambda th: (th["told"], th["t"]))
                   if th["told"] >= first_t - 1e-9)
    # A throw's aim (the monster her attack choice aimed it at: the hunt's attack record's kind, x and platform), told
    # where the tree's step takes one (from 2026-11-02: tracking.MonsterTracker.throw), by place alone: the live track
    # ids are another tracker's
    aims = "aim" in inspect.signature(step.throw).parameters
    asks = {int(k): v for k, v in the_feed.get("asks", {}).items()}

    def read(r):
        if "file" not in r:
            return None
        im = cv2.imread(str(frames / r["file"]))
        return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    out = _Dump(Path(a.out))
    out.write({"head": True, "format": FORMAT, "frames": len(rows), "start": a.start, "map": MAP_ID,
               "character": character.active().id, "t0": t0, "throws": len(throws), "asks": len(asks),
               "feed": hashlib.sha256(Path(a.feed).read_bytes()).hexdigest()[:16],
               "watch": watch is not None})
    side_rows = []
    began_wall = time.time()
    nxt = read(rows[0]) if rows else None
    began = time.monotonic() + 0.5
    told = 0
    for k, r in enumerate(rows):
        full = nxt
        t = t0 + r["t"]
        late = 0.0
        if not a.unpaced:
            due = began + (r["t"] - rows[0]["t"])
            if (wait := due - time.monotonic()) > 0:
                time.sleep(wait)
            late = time.monotonic() - due
        while throws and throws[0]["told"] <= t + 1e-9:
            th = throws.popleft()
            if aims and th.get("x") is not None:
                step.throw(th["t"], th["side"], None, (None, th.get("kind"), th["x"], th.get("platform")))
            else:
                step.throw(th["t"], th["side"])
            told += 1
        if full is None:  # (a gap: the step told the game wasn't on screen, as feed.run does)
            st = step.step(None, t, False, why=r.get("gap", "not in the recording"))
            out.write({"i": r["i"], "t": r["t"], "visible": False, "bad": (st.why or "")[:60]})
            nxt = read(rows[k + 1]) if k + 1 < len(rows) else None
            continue
        ask = asks.get(r["i"])
        use = None if ask is None else worldstate.Asks(t, loot=bool(ask.get("loot")),
                                                       loot_wide=bool(ask.get("loot_wide")),
                                                       mesos_only=bool(ask.get("mesos_only")))
        sights.clear()
        wide.clear()
        said.clear()
        st = step.step(runner.look(Frame(full, t, True, over="")), t, True, use)
        if st.bad is not None:
            runner.recover(st.bad)
        rec = {"i": r["i"], "t": r["t"],
               "her": None if st.where is None else [_r1(st.where[0]), _r1(st.where[1]), st.where[2], st.how,
                                                      st.mode],
               "v": None if st.view is None else [int(st.view[0]), int(st.view[1])], "vs": bool(st.view_sure),
               "bad": None if st.bad is None else st.bad.why[:60],
               "hp": _r3(st.hp), "mp": _r3(st.mp),
               "hpn": None if st.hp_of is None else list(st.hp_of), "lv": st.level_of,
               "m": [[SHORT.get(m.kind, m.kind), _r1(m.x), _r1(m.y), m.platform, m.track,
                      _r3(getattr(m.mob, "fading", None)), m.facing, _r3(getattr(m, "bar", None)), m.mob.x, m.mob.y,
                      m.mob.w, m.mob.h, round(float(m.mob.score), 3), m.mob.frame] for m in st.monsters],
               "tr": [[tr.id, tr.state, SHORT.get(tr.kind, tr.kind), _r1(tr.x), _r1(tr.y),
                       getattr(tr, "platform", None), _r3(getattr(tr, "fading", None)), (tr.how or "")[:1],
                       _r3(getattr(tr, "bar", None)), _r3(getattr(tr, "gain", None)), getattr(tr, "cause", None)]
                      for tr in st.tracks],
               "ev": [[e.kind, e.track, round(e.t - t0, 3), _plain(e.data)] for e in st.events],
               "pl": [[getattr(p, "id", None), getattr(p, "kind", None), _r1(getattr(p, "x", None)),
                       _r1(getattr(p, "y", None)), getattr(p, "platform", None)] for p in (st.players or [])],
               "sg": [[s.point, SHORT.get(s.kind, s.kind), _r1(s.x), _r1(s.y), round(float(s.a), 3),
                       round(float(s.r2), 5), round(float(s.res), 3), int(s.onset), round(s.since - t0, 3)]
                      for s in sights],
               "wd": [[SHORT.get(m.kind, m.kind), _r1(m.foot_x), _r1(m.feet_y), round(float(m.score), 3), m.frame]
                      for m in wide]}
        if st.loot is not None:
            rec["loot"] = [[_r1(d.x), _r1(d.y), d.platform, bool(getattr(d, "mesos", False))] for d in st.loot]
        if said:
            rec["say"] = list(said)
        out.write(rec)
        w = st.work or {}
        side_rows.append([r["i"], round(late * 1000, 2)] + [None if w.get(key) is None else round(w[key] * 1000, 3)
                                                            for key in ("s", "look s", "wide s", "spawns s")])
        nxt = read(rows[k + 1]) if k + 1 < len(rows) else None
    step.close()
    out.write({"end": True, "frames": len(rows), "told": told, "step": step.counts,
               "tracker": dict(step.tracker.counts), "watch": None if watch is None else dict(watch.counts)})
    out.close()
    if a.side:
        lab_dir = root / "gamelab" / "lab"
        reads = {rel: (hashlib.sha256((lab_dir / rel).read_bytes()).hexdigest()[:16]
                       if (lab_dir / rel).is_file() else None) for rel in sorted(opened)}
        Path(a.side).write_text(json.dumps({
            "root": str(root), "frames": str(frames), "pid": os.getpid(), "began": round(began_wall, 3),
            "seconds": round(time.time() - began_wall, 1), "paced": not a.unpaced, "lab_read": reads,
            "times": ["i", "late ms", "pass ms", "look ms", "wide ms", "spawns ms"], "rows": side_rows},
            separators=(",", ":")), encoding="utf-8")
    print(f"replay_run: {len(rows)} frames, {told} throws told, {time.time() - began_wall:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
