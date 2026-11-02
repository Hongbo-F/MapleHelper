"""Replaying the recordings: her own frames through any commit's perception, scored against labels (ROADMAP item 5,
2026-11-02). The simulator can't play the magician (P41), so a perception change for her is checked on the three
lossless recordings of RiceBaby grinding Tree Dungeon on master 40ba190 (R1-R3, 6.6 GB, kept as fixtures from
2026-11-02 by the user's word) instead of by hand-built scripts each time.

    python -m gamelab.sim.replay run [CLIPS] [--root DIR | --commit REF] [--out DIR] [--range A:B] [--unpaced]
                                          [--jobs N]
    python -m gamelab.sim.replay cached [CLIPS] [--commit master] [--range A:B] [--jobs N] [--where]
    python -m gamelab.sim.replay score [CLIPS] [--dumps DIR] [--base master] [--json FILE]
    python -m gamelab.sim.replay check A B          two dumps the same to the byte, or where they part
    python -m gamelab.sim.replay bench [CLIPS] --base REF [--root DIR] [--range A:B] [--rounds 1]
    python -m gamelab.sim.replay feed CLIP          what the hunt's records tell the step (throws, asks, clock)

CLIPS: R1, R2, R3 (by default all three), or their folder names (CLIPS below). Every job that runs frames goes through
the CPU rules' wrapper (heavy.sh / bg_quiet.cmd), as any harness job: a run takes a core and a half while it lasts.

**run**: a code tree's frame step over a recording, exactly as the perception service ran it that night with the
grind attached (gamelab/sim/replay_run.py, a child process per recording that imports the tree's own gamelab:
the service's eyes, senses, step and schedule, her calibration, name tags and models from the tree's lab, the
magician's Shape, 15 frames a second at the frames' own times), fed what the service took from outside from the run's
own records (`feed`): her throws and the drops the hunt asked for. It writes one compact dump a recording (the format
is replay_run's docstring) and a side file with the pass's times. The tree: this checkout as it is (default), another
(`--root`), or a commit (`--commit`: its export under the cache, `tree`, her lab inputs copied from this checkout's).

**cached**: a commit's dumps (master's by default) made once and read after, as the harness's baselines are
(gamelab.sim.baseline): kept under <GAMELAB_REPLAYS, else the baselines' root's recordings/>/<code>/<key>/, `code`
the commit's code_id (its tree under gamelab/, gamedata/, maplehelper/, config.toml, Markdown aside), `key` a hash of
everything else a dump depends on: the runner's own source (replay_run.py), the feed, the lab inputs (LAB_INPUTS, by
content), the GAMELAB_* environment as resolved, Python's and the packages' versions, the recording's frames (stamp)
and the range. A dump is kept only if every lab file it opened is one LAB_INPUTS covers. So a branch is compared
against master without running master again; the labels aren't in the key (a score is made from a dump in seconds,
and `score --base` says which labels' version it read).

**score**: a dump against the recording's labels (tests/fixtures/recordings/<clip>/labels-v2.json: small, versioned,
in git, where no cleanup of the scratch drive can lose them; the frames stay in golden-clips, by path). Per class a
score: the spawn watch's newborns (the onsets of record reproduced, every labelled birth found, false onsets, which
must stay 0, re-onsets, frames to the first sighting), the births it missed (found now?), touches (found by popup or
HP bar, timing), monsters under her body (seen in the state?), her kills (which monster each gain was: the replay's
first and last binding against the label, the first on the victim's level), her casts at her dead (left out by the
state she chose on?), other players (ids, the rival under her dot). `--base master` puts master's cached dumps' scores
beside, from the same labels.

**bench**: pass times, only from runs made side by side: the base and the tree alternated (base, tree, tree, base by
default) on the same frames, paced, one at a time; never against a cached dump's times (the machine changes between
them: the workflows' rule).

The clock: the recordings' frames are stamped in wall time (index.jsonl's 't' from summary.json's 'from', local
UTC-8), the perception service's telemetry in its own time.monotonic for the frame times it names (a victim's gain,
a touch's popup), the hunt's records in wall time at their emission. `clock` fits the one offset that puts the
service's frame times on frames (1793544509.301 s on all three: 106 of 107 in R2 within 2 ms), and the step runs on
the service's own clock (frame 0 at `t0`), so its events read against the live ones directly.
"""

import argparse
import bisect
import datetime
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import baseline

REPO = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("replay_run.py")
GOLDEN = Path(os.environ.get("GAMELAB_GOLDEN_CLIPS", "D:/MapleHelper-scratch/golden-clips"))
LABELS = REPO / "tests" / "fixtures" / "recordings"
LABELS_VERSION = "v2"  # (v1: as the kit was built; v2 from the kill-credit work: README)
MAIN_DATA = Path("D:/MapleHelper/gamedata/data")  # the game data, where a commit's export has none of its own
LOCAL = datetime.timezone(datetime.timedelta(hours=-8))  # the recordings' local time (Pacific, standard)
# The lab files the frame step reads, which git doesn't see (each checkout's copy): her calibration and name tags at
# each size, who plays, her stat window's place (the scene's cover), her plate and pose models (the locator's), the
# map's art (mapart: the camera's registration and the spawn watch). By content; a dump that opened any other lab file
# isn't kept (replay_run's audit: add it here).
LAB_INPUTS = ("character.json", "calibration*.json", "tag_template*.npy", "stat_window.json", "plate*.npz",
              "poses*.npz", "mapart/*.npz", "characters/*/calibration*.json", "characters/*/tag_template*.npy",
              "characters/*/plate*.npz", "characters/*/poses*.npz")
ENV = {"GAMELAB_GPU": "0"}  # the dumps' (as the analyses' and the baselines' replays: the GPU's are the same to the bit)
TURN_S = 0.080  # a throw from the ground that turns her first: its key after the arrow's tap (MOVEMENT.tap_turn_ms 50
                # + actions.TURN_SETTLE_MS 30)
STANCE_S = 1.2  # a throw's side from the fight's stance at most this long before its attack record (kplace's replay)


@dataclass(frozen=True)
class Clip:
    """A recording: its folder in golden-clips, who played, a short name."""
    name: str
    folder: str
    character: str = "magician"

    @property
    def root(self) -> Path:
        return GOLDEN / self.folder

    @property
    def frames(self) -> Path:
        return next(p for p in sorted(self.root.iterdir()) if p.is_dir() and (p / "index.jsonl").exists())

    @property
    def labels(self) -> Path:
        return LABELS / self.folder / f"labels-{LABELS_VERSION}.json"

    def rows(self) -> list[dict]:
        return [json.loads(line) for line in open(self.frames / "index.jsonl", encoding="utf-8")]

    def epoch0(self) -> float:
        """Frame 0's wall time (summary.json's 'from', local)."""
        summary = json.loads((self.frames / "summary.json").read_text(encoding="utf-8"))
        return datetime.datetime.strptime(summary["from"], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=LOCAL).timestamp()

    def telemetry(self, what: str) -> Path:
        return next((self.root / "telemetry").glob(f"*-{what}.jsonl"))


# The three of 2026-11-02 (RiceBaby, level 31-32, master 40ba190; README.json in each folder says what it is)
CLIPS = {c.name: c for c in (Clip("R1", "ricebaby-grind-20261102-0009"),
                             Clip("R2", "ricebaby-grind-20261102-0017-5min"),
                             Clip("R3", "ricebaby-grind-20261102-0017-5min-b"))}


def clip_of(name: str) -> Clip:
    for c in CLIPS.values():
        if name in (c.name, c.folder):
            return c
    raise SystemExit(f"replay: no recording {name!r} (one of {', '.join(CLIPS)})")


def _records(path: Path, lo: float, hi: float, topics: set[str]) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"topic"' not in line:
                continue
            r = json.loads(line)
            if r.get("topic") in topics and lo <= r.get("t", 0.0) <= hi:
                out.append(r)
    return out


def clock(clip: Clip) -> dict:
    """The offset (wall time - the service's time.monotonic) that puts the frame times the service's telemetry names
    (victims' gains, touches' popups) on the recording's frames: the one, on a 1 ms grid of the candidates, that puts
    the most of them within 2 ms of a frame; with how many it puts there of how many."""
    rows = clip.rows()
    e0 = clip.epoch0()
    ep = [e0 + r["t"] for r in rows]
    named = []
    for r in _records(clip.telemetry("perceive"), ep[0] - 5, ep[-1] + 5, {"event"}):
        if r.get("kind") == "victim" and "gain " in r.get("text", ""):
            named.append((r["t"], float(r["text"].split("gain ")[1].split(",")[0])))
        elif r.get("kind") == "touch" and r.get("at") is not None:
            named.append((r["t"], float(r["at"])))
    if not named:
        raise SystemExit(f"replay: {clip.name}'s perceive telemetry names no frame times")
    rough = sorted(e - m for e, m in named)[len(named) // 2]
    cands = []
    for _, m in named:
        lo, hi = bisect.bisect_left(ep, m + rough - 1.0), bisect.bisect_right(ep, m + rough + 0.2)
        cands.append([ep[i] - m for i in range(lo, hi)])
    best = (-1, rough)
    for o in sorted({round(x, 3) for c in cands for x in c}):
        n = sum(any(abs(x - o) <= 0.002 for x in c) for c in cands)
        if n > best[0]:
            best = (n, o)
    return {"offset": best[1], "matched": best[0], "of": len(named), "epoch0": e0, "t0": e0 - best[1]}


def feed(clip: Clip) -> dict:
    """What the perception service took from outside while the grind ran, from the run's own records: frame 0's time
    on the service's clock (`t0`, by `clock`), her throws (the tour's `attack` records: each one key, with the monster
    her attack choice aimed it at, its kind, x and platform, and its track where the record says (from 2026-11-02; the
    runner tells the aim by place: replay_run); its side as the record says, else from the fight's `stance` just before
    it, else her target's side of her by the last scene; its key TURN_S later where her
    keys didn't have her facing that way, as Actions.throw taps the arrow first; `told` when the record went, which is
    when the service heard of it) and the drops the hunt asked for (each tour `scene` that looked at drops: an ask on
    the last frame shown 12 ms before it, its platform's or every one in view as the look was wide, mesos alone if her
    backpack was full by the `drops` record beside it). Plain data, as replay_run reads it, with counts of how each was
    told (`how`)."""
    c = clock(clip)
    rows = clip.rows()
    e0, off = c["epoch0"], c["offset"]
    lo, hi = e0 - 30.0, e0 + rows[-1]["t"] + 1.0
    tour = _records(clip.telemetry("tour"), lo, hi, {"attack", "stance", "scene", "drops"})
    how = {"attacks": 0, "side by stance": 0, "side by place": 0, "no side": 0, "turned": 0}
    throws, facing = [], (None, -1e18)  # (her keys' way, as of when)
    stance, scene, full = None, None, False
    for r in tour:
        topic = r["topic"]
        if topic == "stance":
            stance = r
        elif topic == "drops":
            full = bool(r.get("inventory_full"))
        elif topic == "scene":
            scene = r
            her = r.get("her") or {}
            keys = her.get("facing_keys")
            if keys in ("left", "right") and r["t"] >= facing[1]:
                facing = (keys, r["t"])
        elif topic == "attack":
            how["attacks"] += 1
            side = r.get("side") if r.get("side") in ("left", "right") else None  # (told from 2026-11-02)
            if side is not None:
                how["side told"] = how.get("side told", 0) + 1
            if side is None and stance is not None and r["t"] - stance["t"] <= STANCE_S                     and stance.get("act") == "throw":
                m = re.search(r"\b(left|right)\b", stance.get("text", "").split("(")[0])
                side = m[1] if m else None
                how["side by stance"] += side is not None
            if side is None and scene is not None and r.get("x") is not None and (scene.get("her") or {}).get("x") \
                    is not None:
                side = "right" if r["x"] >= scene["her"]["x"] else "left"
                how["side by place"] += 1
            if side is None:
                how["no side"] += 1
                continue
            her = (scene or {}).get("her") or {}
            faces = facing[0] == side or (her.get("facing_seen") == side and her.get("facing_strong"))
            turn = 0.0 if faces else TURN_S
            how["turned"] += turn > 0
            told = r["t"] - off
            throws.append({"t": round(told + turn, 4), "side": side, "told": round(told, 4),
                           "kind": r.get("kind"), "x": r.get("x"), "platform": r.get("platform"), "key": r.get("key"),
                           **({"track": r["track"]} if r.get("track") is not None else {})})
            facing = (side, r["t"])
    ep = [e0 + r["t"] for r in rows]
    asks = {}
    for r in tour:
        if r["topic"] != "scene" or not r.get("looked"):
            continue
        k = bisect.bisect_right(ep, r["t"] - 0.012) - 1
        if 0 <= k < len(rows):
            wide = bool(r.get("wide"))
            asks[str(rows[k]["i"])] = {"loot": not wide, "loot_wide": wide, "mesos_only": False}
    for r in tour:  # (her backpack full: the asks for mesos alone, as the drops record at the same look says)
        if r["topic"] == "drops" and r.get("inventory_full"):
            k = bisect.bisect_right(ep, r["t"] - 0.012) - 1
            if 0 <= k < len(rows) and str(rows[k]["i"]) in asks:
                asks[str(rows[k]["i"])]["mesos_only"] = True
    how["asks"] = len(asks)
    return {"clip": clip.folder, "t0": round(c["t0"], 4), "clock": c, "throws": throws, "asks": asks, "how": how}


# --- the trees run -------------------------------------------------------------------------------------------------

def cache_root() -> Path:
    return Path(os.environ.get("GAMELAB_REPLAYS", str(baseline.ROOT / "recordings")))


def lab_inputs(lab: Path) -> dict:
    """LAB_INPUTS under `lab`: {relative path: 16 hex of its content}, character.json None where it's absent."""
    out = {"character.json": None}
    for pattern in LAB_INPUTS:
        for p in sorted(lab.glob(pattern)):
            if p.is_file():
                out[p.relative_to(lab).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return dict(sorted(out.items()))


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def tree(ref: str, lab_from: Path = REPO / "gamelab" / "lab") -> tuple[Path, str, str]:
    """Commit `ref`'s code as a tree to run (git archive of gamelab/, gamedata/, maplehelper/, config.toml), under
    the cache's trees/<code 12>-<lab 8>/, with the DLLs this checkout has built (native names them by their sources'
    hash: a source that differs is built there on first use) and the lab inputs of `lab_from` (by default this
    checkout's: both sides of a comparison read the same). Made once; (the folder, the commit, its code)."""
    sha = baseline.rev(ref, REPO)
    code = baseline.code_id(sha, REPO)
    inputs = lab_inputs(lab_from)
    dest = cache_root() / "trees" / f"{code[:12]}-{_hash(inputs)[:8]}"
    if (dest / ".ready").exists():
        return dest, sha, code
    tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
    tmp.mkdir(parents=True)
    tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", sha, "--", *baseline.CODE],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(tmp, filter="data")
    build = REPO / "gamelab" / "build"
    if build.is_dir():
        (tmp / "gamelab" / "build").mkdir(exist_ok=True)
        for dll in build.glob("*.dll"):
            shutil.copy2(dll, tmp / "gamelab" / "build" / dll.name)
    for rel, digest in inputs.items():
        if digest is not None:
            (tmp / "gamelab" / "lab" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lab_from / rel, tmp / "gamelab" / "lab" / rel)
    (tmp / ".ready").write_text(json.dumps({"commit": sha, "code": code, "lab": inputs}, indent=1), encoding="utf-8")
    os.replace(tmp, dest)
    return dest, sha, code


def frames_stamp(clip: Clip) -> str:
    """The recording's frames: its index's content and every file's name and size."""
    h = hashlib.sha256((clip.frames / "index.jsonl").read_bytes())
    for e in sorted(os.scandir(clip.frames), key=lambda e: e.name):
        if e.is_file():
            h.update(f"{e.name}\0{e.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


def _range(text: str) -> tuple[int, int]:
    if not text:
        return 0, 0
    a, _, b = text.partition(":")
    start = int(a or 0)
    return start, (int(b) - start if b else 0)


def run_one(clip: Clip, root: Path, out: Path, start: int = 0, limit: int = 0, paced: bool = True,
            the_feed: dict | None = None) -> subprocess.Popen:
    """replay_run.py on `clip` with tree `root`, writing out/<folder>.jsonl.gz and .side.json (started: the process)."""
    out.mkdir(parents=True, exist_ok=True)
    the_feed = feed(clip) if the_feed is None else the_feed
    feed_path = out / f"{clip.folder}.feed.json"
    feed_path.write_text(json.dumps(the_feed, separators=(",", ":")), encoding="utf-8")
    data = [] if (root / "gamedata" / "data" / "out").exists() else ["--data", str(MAIN_DATA)]
    cmd = [sys.executable, "-B", str(RUNNER), "--root", str(root), "--frames", str(clip.frames), "--feed",
           str(feed_path), "--out", str(out / f"{clip.folder}.jsonl.gz"), "--side",
           str(out / f"{clip.folder}.side.json"), "--start", str(start), "--limit", str(limit), *data] \
        + ([] if paced else ["--unpaced"])
    env = dict(os.environ, GAMELAB_CHARACTER=clip.character, **ENV)
    log = open(out / f"{clip.folder}.log", "w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)


def run_all(clips: list[Clip], root: Path, out: Path, start: int, limit: int, paced: bool, jobs: int) -> list[Path]:
    """Each recording through `root`, `jobs` at once; the dumps' paths (a failed one said, and left out)."""
    pending, running, done = list(clips), [], []
    while pending or running:
        while pending and len(running) < max(1, jobs):
            c = pending.pop(0)
            running.append((c, run_one(c, root, out, start, limit, paced)))
        time.sleep(0.5)
        for c, p in list(running):
            if p.poll() is not None:
                running.remove((c, p))
                dump = out / f"{c.folder}.jsonl.gz"
                if p.returncode == 0 and dump.exists():
                    done.append(dump)
                else:
                    tail = (out / f"{c.folder}.log").read_text(encoding="utf-8", errors="replace")[-600:]
                    print(f"replay: {c.name} failed (exit {p.returncode}):\n{tail}", flush=True)
    return done


def key_of(clip: Clip, code: str, lab: dict, start: int, limit: int, the_feed: dict) -> dict:
    """What a cached dump is of, beside its code (cached)."""
    return {"format": 1, "code": code, "runner": hashlib.sha256(RUNNER.read_bytes()).hexdigest()[:16],
            "feed": _hash(the_feed)[:16], "lab": _hash(lab)[:16], "env": dict(ENV, GAMELAB_CHARACTER=clip.character),
            "python": sys.version.split()[0], "packages": baseline.packages(), "frames": frames_stamp(clip),
            "range": [start, limit]}


def covered(read: dict) -> list[str]:
    """Of the lab files a dump opened, those LAB_INPUTS doesn't cover."""
    import fnmatch
    return [rel for rel in read if not any(fnmatch.fnmatch(rel, pat) for pat in LAB_INPUTS)]


def cached(clips: list[Clip], ref: str, start: int, limit: int, jobs: int, make: bool = True) -> dict[str, Path]:
    """Commit `ref`'s dumps of `clips` (see the top): read where they're kept, else made (paced) and kept. {clip's
    folder: its dump}; one that can't be had is said and left out."""
    sha = baseline.rev(ref, REPO)
    code = baseline.code_id(sha, REPO)
    lab = lab_inputs(REPO / "gamelab" / "lab")
    out, todo = {}, []
    for c in clips:
        f = feed(c)
        k = key_of(c, code, lab, start, limit, f)
        where = cache_root() / code[:12] / _hash(k)[:16]
        dump = where / f"{c.folder}.jsonl.gz"
        if dump.exists():
            out[c.folder] = dump
        else:
            todo.append((c, where, k))
    if todo and make:
        root, _, _ = tree(ref)
        pending, running = list(todo), []
        while pending or running:  # (`jobs` recordings at once, each into its key's folder)
            while pending and len(running) < max(1, jobs):
                c, where, k = pending.pop(0)
                tmp = where.with_name(where.name + f".tmp{os.getpid()}")
                running.append((c, where, k, tmp, run_one(c, root, tmp, start, limit, True)))
            time.sleep(0.5)
            for item in list(running):
                c, where, k, tmp, p = item
                if p.poll() is None:
                    continue
                running.remove(item)
                if p.returncode != 0 or not (tmp / f"{c.folder}.jsonl.gz").exists():
                    tail = (tmp / f"{c.folder}.log").read_text(encoding="utf-8", errors="replace")[-600:]
                    print(f"replay: {c.name} failed (exit {p.returncode}):\n{tail}", flush=True)
                    continue
                if (got := _keep(c, where, k, tmp, sha)) is not None:
                    out[c.folder] = got
    elif todo:
        for c, where, _ in todo:
            print(f"not cached: {c.name} ({where})")
    return out


def _keep(c: Clip, where: Path, k: dict, tmp: Path, sha: str) -> Path | None:
    """A dump made for the cache (in `tmp`) kept under its key's folder, if every lab file it opened is keyed."""
    side = json.loads((tmp / f"{c.folder}.side.json").read_text(encoding="utf-8"))
    if bad := covered(side["lab_read"]):
        print(f"replay: {c.name}'s dump opened lab files LAB_INPUTS doesn't key ({', '.join(bad)}): not kept (it's in "
              f"{tmp})")
        return None
    where.mkdir(parents=True, exist_ok=True)
    for p in tmp.iterdir():
        os.replace(p, where / p.name)
    tmp.rmdir()
    (where / "key.json").write_text(json.dumps(dict(k, commit=sha, made=time.strftime("%Y-%m-%d %H:%M")), indent=1),
                                    encoding="utf-8")
    return where / f"{c.folder}.jsonl.gz"


# --- reading dumps -------------------------------------------------------------------------------------------------

def load(path: Path) -> tuple[dict, list[dict], dict]:
    """(head, frames in order, end) of a dump."""
    head, frames, end = {}, [], {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("head"):
                head = r
            elif r.get("end"):
                end = r
            else:
                frames.append(r)
    return head, frames, end


def check(a: Path, b: Path) -> int:
    """Whether two dumps are the same to the byte (0), else the first frame where they part and in what (1)."""
    if Path(a).read_bytes() == Path(b).read_bytes():
        print(f"the same to the byte: {a} and {b}")
        return 0
    ha, fa, ea = load(a)
    hb, fb, eb = load(b)
    if ha != hb:
        print(f"heads differ: {[k for k in set(ha) | set(hb) if ha.get(k) != hb.get(k)]}")
    for x, y in zip(fa, fb):
        if x != y:
            keys = [k for k in set(x) | set(y) if x.get(k) != y.get(k)]
            print(f"first differs at frame {x.get('i')}: {', '.join(sorted(keys))}")
            break
    else:
        if len(fa) != len(fb):
            print(f"frames: {len(fa)} against {len(fb)}")
        elif ea != eb:
            print("the end lines differ")
    return 1


def bench(clips: list[Clip], base: str, root: Path, out: Path, start: int, limit: int, rounds: int) -> int:
    """The pass's times of `base` and `root`, alternated on the same frames (base, tree, tree, base per round), paced,
    one at a time: per side the median, p90 and p99 of the pass and of the spawn watch, over every run of it."""
    base_root, _, _ = tree(base)
    order = []
    for _ in range(rounds):
        order += [("base", base_root), ("tree", root), ("tree", root), ("base", base_root)]
    times: dict[str, dict[str, list[float]]] = {"base": {}, "tree": {}}
    for n, (side, r) in enumerate(order, 1):
        got = run_all(clips, r, out / f"{n}-{side}", start, limit, True, 1)
        for dump in got:
            s = json.loads(dump.with_name(dump.name.replace(".jsonl.gz", ".side.json")).read_text(encoding="utf-8"))
            for row in s["rows"]:
                for name, v in zip(("pass", "look", "wide", "spawns"), row[2:]):
                    if v is not None:
                        times[side].setdefault(name, []).append(v)
        print(f"  {n}/{len(order)} {side} done", flush=True)
    for side in ("base", "tree"):
        for name, v in sorted(times[side].items()):
            v = sorted(v)
            print(f"{side:5} {name:7} median {statistics.median(v):6.2f} ms  p90 {v[int(0.9 * (len(v) - 1))]:6.2f}  "
                  f"p99 {v[int(0.99 * (len(v) - 1))]:6.2f}  ({len(v)} frames)")
    return 0


# --- the command line -----------------------------------------------------------------------------------------------

def _clips(names: list[str]) -> list[Clip]:
    return [clip_of(n) for n in names] if names else list(CLIPS.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gamelab.sim.replay", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="a tree's frame step over the recordings")
    r.add_argument("clips", nargs="*")
    r.add_argument("--root", type=Path, help="a checkout to run as it is (default: this one)")
    r.add_argument("--commit", help="a commit to run (its export, made once under the cache)")
    r.add_argument("--out", type=Path)
    r.add_argument("--range", default="")
    r.add_argument("--unpaced", action="store_true")
    r.add_argument("--jobs", type=int, default=3)
    c = sub.add_parser("cached", help="a commit's dumps, made once and read after")
    c.add_argument("clips", nargs="*")
    c.add_argument("--commit", default="master")
    c.add_argument("--range", default="")
    c.add_argument("--jobs", type=int, default=3)
    c.add_argument("--where", action="store_true", help="only say where they are; make none")
    s = sub.add_parser("score", help="dumps against the recordings' labels")
    s.add_argument("clips", nargs="*")
    s.add_argument("--dumps", type=Path, help="a run's folder (default: --base's cached dumps alone)")
    s.add_argument("--base", default="", help="a commit whose cached dumps are scored beside (made if missing)")
    s.add_argument("--json", type=Path)
    k = sub.add_parser("check", help="two dumps the same to the byte?")
    k.add_argument("a", type=Path)
    k.add_argument("b", type=Path)
    b = sub.add_parser("bench", help="pass times, base and tree alternated")
    b.add_argument("clips", nargs="*")
    b.add_argument("--base", default="master")
    b.add_argument("--root", type=Path, default=REPO)
    b.add_argument("--out", type=Path)
    b.add_argument("--range", default="")
    b.add_argument("--rounds", type=int, default=1)
    f = sub.add_parser("feed", help="what the hunt's records tell the step")
    f.add_argument("clip")
    a = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if a.cmd == "feed":
        got = feed(clip_of(a.clip))
        print(json.dumps({k: got[k] for k in ("clip", "t0", "clock", "how")}, indent=1))
        return 0
    if a.cmd == "check":
        return check(a.a, a.b)
    if a.cmd == "run":
        if a.commit:
            root, sha, code = tree(a.commit)
            print(f"replay: {a.commit} = {sha[:10]} (code {code[:12]}) at {root}")
        else:
            root = (a.root or REPO).resolve()
        start, limit = _range(a.range)
        out = a.out or cache_root() / "runs" / stamp
        got = run_all(_clips(a.clips), root, out, start, limit, not a.unpaced, a.jobs)
        for p in got:
            print(p)
        return 0 if len(got) == len(_clips(a.clips)) else 1
    if a.cmd == "cached":
        start, limit = _range(a.range)
        got = cached(_clips(a.clips), a.commit, start, limit, a.jobs, make=not a.where)
        for p in got.values():
            print(p)
        return 0 if len(got) == len(_clips(a.clips)) else 1
    if a.cmd == "bench":
        start, limit = _range(a.range)
        return bench(_clips(a.clips), a.base, a.root.resolve(), a.out or cache_root() / "bench" / stamp, start, limit,
                     a.rounds)
    if a.cmd == "score":
        from . import replay_score
        return replay_score.main(a, _clips(a.clips))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
