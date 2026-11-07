"""The web UI's review pages (gamelab/ui/review.py) of the Lie Detector mini-game (ld) and the Violetta game (violetta),
from the alarm clips: the one command, from any folder. README.md says what each group shows and what stays out of git.

usage: build.py {ld,violetta} [clip keys ...] [--missing] [--list] [--root DIR] [--force] [--weights FILE]
  a clip's key is its stamp, 20261104-180343 (its whole folder name will do)
  --list     which clips of that form have no page in each group; builds nothing
  --missing  build those (ld: a clip with no range is skipped, saying why)
  --force    build the named clips' pages again, in place of the ones there; without it a page is never written over
  --root     where the pages go (default: where the web UI looks, GAMELAB_REVIEW_ROOT or its own default)
  --weights  the Violetta detector (default: gamelab/assets/violetta.pt)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

CLIPS = common.paths.ALARM_CLIPS
FORMS = {"ld": "*mini-game-notice*", "violetta": "*countdown*"}  # the alarm clips each game is on
REFERENCE = "ld-reference"
GROUPS = {"ld": (REFERENCE, "ld-outline-pf3", "ld-speed-compare"), "violetta": ("violetta", "violetta-yolo")}
PF3 = dict(mu=1.0, track=0.0, reseed=0.0, gamma=3.0, lam=0.3, offboard=1.0, turn=0.1, turn_edge=0.3)  # ld-outline-pf3's
WORKERS = 4


def clips_of(form):
    """The form's alarm clips: key -> folder name."""
    return {d.name[:15]: d.name for d in sorted(CLIPS.glob(FORMS[form])) if d.is_dir()}


def pages(root, group, key):
    """A clip's page in a group, if it has one: <root>/<group>/<key>-<fps>fps."""
    return [d for d in (root / group).glob(f"{key}-*fps") if d.is_dir()]


def missing(form, keys, root):
    """Each group's clips with no page there."""
    return {g: [k for k in keys if not pages(root, g, k)] for g in GROUPS[form]}


def plan(form, keys, root, force=False, approved=lambda key: False):
    """Clip -> the groups to build, saying what is left out and why: a page that is there, unless forced; an approved
    clip's reference, ever."""
    todo = {}
    for key in keys:
        for g in GROUPS[form]:
            if g == REFERENCE and approved(key):
                print(f"{key}: {g} skipped: its reference is approved, never rebuilt", flush=True)
            elif pages(root, g, key) and not force:
                print(f"{key}: {g} skipped: its page is there (--force to build it again)", flush=True)
            else:
                todo.setdefault(key, []).append(g)
    return todo


def clear(todo, root):
    """--force: the pages about to be built again go first (the builders never write over one)."""
    for key, groups in todo.items():
        for g in groups:
            for page in pages(root, g, key):
                print(f"{key}: {g}: its page goes, to be built again", flush=True)
                shutil.rmtree(page)


def build_ld(todo):
    from concurrent.futures import ProcessPoolExecutor

    import lib
    import reference
    import review_compare
    import review_pf3
    from scrolls import scrolls
    for k in todo:  # (here first: the workers would each make these, and read each other's half-written files)
        lib.load_clip(k)
        scrolls(k)
    need = [k for k, groups in todo.items() if not lib.reference_of(k)[1]
            and (REFERENCE in groups or not lib.reference_of(k)[0].exists())]
    runs = {k: {} for k in need}
    with ProcessPoolExecutor(WORKERS) as ex:
        for key, seed, res, j0 in ex.map(reference.one, [(k, s) for k in need for s in reference.SEEDS]):
            runs[key][seed] = res
            if len(runs[key]) == len(reference.SEEDS):
                reference.build(key, runs[key], page=REFERENCE in todo[key])
    for k, groups in todo.items():
        if "ld-outline-pf3" in groups:
            review_pf3.render(k, "ld-outline-pf3", PF3)
        if "ld-speed-compare" in groups:
            review_compare.render(k)


def run(script, *args, **kw):
    """A step of the study's, in its own process: found by its place, run in the working folder."""
    print(">", script, *args, flush=True)
    return subprocess.run([sys.executable, "-X", "utf8", str(HERE / "violetta" / script), *args], check=True, **kw)


def tally(dets, names):
    """The rules solver on the rounds against the answers given: its lines, and its last (the tally) kept."""
    lines = run("rules.py", "--dets", dets, *names, stdout=subprocess.PIPE, text=True).stdout.strip()
    print(lines, flush=True)
    return lines.splitlines()[-1]


def build_violetta(todo, weights):
    """Each step skipped where its output is there. All in <work>/violetta, the study's scripts' relative folders: first
    pass dets/, solved games solved/, sprite matches dets5/, YOLO's detsY/."""
    import numpy as np
    from PIL import Image

    work = common.work() / "violetta"
    for sub in ("dets", "dets5", "detsY", "solved"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    import solve
    from gamelab.perception.lie_detector import LieDetectorSpotter
    spotter = None
    for name in todo:
        clip = CLIPS / name
        first = Path("dets") / f"{name}.jsonl"
        if not first.exists():
            # from 14 s after the alarm (the 10 s box, the window coming) to a second after its last sight in the clip
            spotter = spotter or LieDetectorSpotter()
            s = json.load(open(clip / "summary.json", encoding="utf-8"))
            rows = [json.loads(line) for line in open(clip / "index.jsonl", encoding="utf-8")]
            frames = [r for r in rows if "file" in r]
            last = max(r["t"] for r in frames[::3]
                       if spotter.near(np.asarray(Image.open(clip / r["file"]).convert("RGB")))["countdown"] >= 1)
            run("track2.py", str(clip), str(s["alarm_t"] + 14), str(last + 1), f"{first}.part",
                "--workers", str(WORKERS))
            Path(f"{first}.part").rename(first)
        solved = Path("solved") / f"{name}.json"
        if not solved.exists():
            res = solve.main(str(clip), str(first))
            print(f"{name}: made-up at {res['made_up_start']} (votes {res['start_votes']}), answers {res['answers']}",
                  flush=True)
            solved.write_text(json.dumps(res, indent=1))
        matches = Path("dets5") / f"{name}.jsonl"
        if not matches.exists():
            run("track5.py", str(clip), f"{matches}.part", "--workers", str(WORKERS))
            Path(f"{matches}.part").rename(matches)
    sprite = [n for n, groups in todo.items() if "violetta" in groups]
    yolo = [n for n, groups in todo.items() if "violetta-yolo" in groups]
    boxes = [n for n in yolo if not (Path("detsY") / f"{n}.jsonl").exists()]
    if boxes:
        # (into a folder of its own, as the steps above their .part: a crash leaves no half a file taken for done)
        run("yolo_detect.py", weights, "detsY.part", *boxes)
        for n in boxes:
            (Path("detsY.part") / f"{n}.jsonl").rename(Path("detsY") / f"{n}.jsonl")
    tallies = {}
    if sprite:
        tallies["sprite matches"] = tally("dets5", sprite)
    if yolo:
        tallies["YOLO"] = tally("detsY", yolo)
    if sprite:
        run("build_review.py", *sprite)
    if yolo:
        run("build_review.py", "--dets", "detsY", "--group", "violetta-yolo", *yolo)
    for source, line in tallies.items():
        print(f"{source}: {line}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("form", choices=GROUPS)
    p.add_argument("keys", nargs="*", metavar="clip")
    p.add_argument("--missing", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--root")
    p.add_argument("--force", action="store_true")
    p.add_argument("--weights", default=str(common.WEIGHTS))
    a = p.parse_intermixed_args(argv)
    root = Path(a.root or common.root()).resolve()
    os.environ["GAMELAB_REVIEW_ROOT"] = str(root)  # (the builders read it at their import, in workers and steps too)
    clips = clips_of(a.form)
    if a.list:
        print(f"{a.form}: {len(clips)} clips, pages under {root}")
        for g, keys in missing(a.form, clips, root).items():
            print(f"{g}: {len(keys)} with no page: {' '.join(keys) or '-'}")
        return
    if a.missing == bool(a.keys):
        p.error("name the clips, or --missing, or --list")
    if a.force and not a.keys:
        p.error("--force is for the clips named")
    keys = [k[:15] for k in a.keys]
    if a.missing:
        keys = sorted({k for ks in missing(a.form, clips, root).values() for k in ks})
    unknown = [k for k in keys if k not in clips]
    if unknown:
        p.error(f"no {a.form} clip: {' '.join(unknown)}")
    sys.path.insert(0, str(HERE / a.form))
    if a.form == "ld":
        import lib
        todo = plan(a.form, keys, root, a.force, approved=lambda k: lib.reference_of(k)[1])
        todo = {k: groups for k, groups in todo.items() if lib.clip(k) is not None}  # (no range: it has said why)
    else:
        todo = plan(a.form, keys, root, a.force)
    if a.force:
        clear(todo, root)
    if a.form == "ld":
        build_ld(todo)
    else:
        build_violetta({clips[k]: groups for k, groups in todo.items()}, str(Path(a.weights).resolve()))


if __name__ == "__main__":
    main()
