"""The baseline cache: the harness's results a seed a file, kept by the code's identity and the run's arguments, so the
side of a comparison that doesn't change (master's) is run once and read after (sim.harness: `run --cache`, `cached`,
`ab`; README, Checking a change).

    <ROOT>/harness/<code>/<key>/seed-NNN.json   one seed's result: the dict harness.run_one gives (run_many's per seed)
    <ROOT>/harness/<code>/<key>/meta.json       what the key covers, readably, and the commit that ran it first

ROOT is D:/MapleHelper-scratch/baselines (GAMELAB_BASELINES says otherwise). Only deterministic results go in: a run of
the same code with the same arguments on the same seed gives the same dict to the digit (the harness's runs are
PYTHONHASHSEED=0, her stats fixed, the world and the policy seeded); `think` charges this computer's own computing to
the clock (fakes.Clock), so a run with it isn't cached. Timing benches never are: what they measure is the computer.

The code's identity (`identity`, `code_id`) is its commit's code: a SHA-1 of the commit's tree under gamelab/,
gamedata/, maplehelper/ and config.toml (every file's mode, blob and path; the Markdown, which nothing runs, left out),
so commits that change only tests, READMEs or files elsewhere share one baseline (master's README-only and test-only
commits don't run it again), and a commit is named the same however it's asked for (`master`, a SHA). Only while the
checkout's code is as committed: no change, staged or not, and no untracked file there (git status; files git ignores
aside, such as gamelab/lab, gamelab/build and gamedata/data). Otherwise nothing is read or kept (a checkout being worked
on runs everything): a hash of the working tree could key those too, but they change with every edit, so it would keep
runs nobody reads again; what's asked for again is a commit's ("master"). The DLLs (gamelab/build, ignored) are built
from native.c and search.c and named by their sources' hash (native.dll_path), so the code names them too.

The key (`key`: 16 hex of a SHA-256) is of:
  - every argument of harness.run_one but the seed, bound with its defaults (`params`): those of CORE always, any other
    only where it isn't at its default, so a switch a later commit adds, off, leaves the key of every run before it as
    it was (and a base and a candidate whose only difference is a new switch, off, pair up);
  - the game data, which isn't in git (`data_stamp`: every file under gamedata/data by its path, size and mtime: a file
    extracted again changes the key, and nothing is read from a stamp that can't be made);
  - Python's version and the installed packages' a run imports (`packages`: numpy, OpenCV, Pillow, mss; every one of
    the distributions that could give its module, so two OpenCV wheels side by side are both in it; tests/
    test_baseline.py checks that a run imports no other);
  - the GAMELAB_* environment as what it resolves to (`gamelab_env`): the character played (character.active_id) and
    whether her stats are fixed, the GPU mode (native.gpu_mode: GAMELAB_GPU, else lab/GPU), and any other GAMELAB_*
    variable as it is (none changes a run today, so one set splits the cache rather than risk it); where the cache is
    and who fills it (NOT_KEYED: GAMELAB_BASELINES, GAMELAB_BASE_FILL) aside;
  - FORMAT, this scheme's version.
`ab` hands the base's filler the environment its key resolved (`filler_env`), so the base's seeds land where it reads.
The DLLs' compiler isn't in it: they're built with /fp:precise to give Python's answers to the bit (native.FLAGS), and
the tests check that they do.
A seed's file is written whole (a temporary file, then renamed), so a run cut short leaves the seeds it finished.
"""

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
from pathlib import Path

FORMAT = 2  # 2: the installed packages' versions and the GAMELAB_* environment as resolved (1: numpy's, as set)
ROOT = Path(os.environ.get("GAMELAB_BASELINES", "D:/MapleHelper-scratch/baselines"))
REPO = Path(__file__).resolve().parents[2]
CODE = ("gamelab", "gamedata", "maplehelper", "config.toml")  # what identity() requires as committed
NOT_CODE = (":(exclude,glob)**/*.md",)                      # ... but these (nothing runs them)
CORE = ("policy", "preset", "minutes", "think", "costs", "ways", "map_id", "world", "grounds", "attack")
UNCACHED = {"think": 0.0}  # arguments whose other values aren't deterministic, and so not cached
# The installed packages a run imports, by module, and the distributions that could give each (OpenCV's four wheels:
# this env has two, opencv-python and opencv-python-headless, either one's files cv2's).
PACKAGES = {"numpy": ("numpy",),
            "cv2": ("opencv-python", "opencv-python-headless", "opencv-contrib-python",
                    "opencv-contrib-python-headless"),
            "PIL": ("pillow",), "mss": ("mss",)}
NOT_KEYED = ("GAMELAB_BASELINES", "GAMELAB_BASE_FILL")  # where the cache is and who fills it, not what a run does
RESOLVED = ("GAMELAB_CHARACTER", "GAMELAB_FIXED_STATS", "GAMELAB_GPU")  # keyed as what they resolve to (gamelab_env)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, stdin=subprocess.DEVNULL)


def identity(repo: Path = REPO) -> tuple[str | None, str]:
    """(the checkout's code: code_id of HEAD, "") while its code (CODE) is as committed; else (None, why not)."""
    head = _git(repo, "rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        return None, f"not a git checkout with a commit ({head.stderr.strip() or head.returncode})"
    status = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *CODE, *NOT_CODE)
    if status.returncode != 0:
        return None, f"git status failed: {status.stderr.strip()}"
    changed = [line[3:] for line in status.stdout.splitlines() if line.strip()]
    if changed:
        shown = ", ".join(changed[:4]) + (f" and {len(changed) - 4} more" if len(changed) > 4 else "")
        return None, f"the checkout's code isn't as committed ({shown})"
    return code_id(head.stdout.strip(), repo), ""


def code_id(commit: str, repo: Path = REPO) -> str:
    """A commit's code: a SHA-1 of its tree's entries under CODE (git ls-tree: mode, blob, path), the Markdown aside."""
    tree = _git(repo, "ls-tree", "-r", "--full-tree", commit, "--", *CODE)
    if tree.returncode != 0:
        raise ValueError(f"no tree for {commit!r} ({tree.stderr.strip()})")
    lines = [line for line in tree.stdout.splitlines() if line.strip() and not line.endswith(".md")]
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()


def params(fn, args: tuple = (), kwargs: dict | None = None) -> dict:
    """The arguments of a call of `fn` (harness.run_one) as the key has them: bound by name with its defaults, the seed
    left out; CORE's always, any other only where it isn't at its default; tuples as lists, and a whole number where
    the default is a float as a float (5 and 5.0 minutes are one run)."""
    sig = inspect.signature(fn)
    bound = sig.bind(*args, **(kwargs or {}))
    bound.apply_defaults()
    out = {}
    for name, value in bound.arguments.items():
        if name == "seed":
            continue
        default = sig.parameters[name].default
        if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if isinstance(value, tuple):
            value = list(value)
        if name not in CORE and value == (list(default) if isinstance(default, tuple) else default):
            continue
        out[name] = value
    return out


def uncacheable(p: dict) -> str | None:
    """Why a run with these arguments isn't cached (not deterministic), or None."""
    for name, only in UNCACHED.items():
        if p.get(name, only) != only:
            return f"{name} {p[name]} isn't deterministic (it charges this computer's computing to the clock)"
    return None


def data_stamp(repo: Path = REPO) -> str:
    """The game data's stamp: every file under gamedata/data (through its junctions) by path, size and mtime."""
    root = repo / "gamedata" / "data"
    h = hashlib.sha256()
    for path, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(path, name)
            st = os.stat(full)
            h.update(f"{os.path.relpath(full, root)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def packages() -> dict:
    """The versions of the distributions that could give the modules a run imports (PACKAGES), None if not installed."""
    out = {}
    for dists in PACKAGES.values():
        for dist in dists:
            try:
                out[dist] = importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                out[dist] = None
    return out


def gamelab_env(environ=None) -> dict:
    """The GAMELAB_* environment as it bears on a run: the character played and whether her stats are fixed
    (character.active_id: GAMELAB_CHARACTER, under GAMELAB_FIXED_STATS the thief by default), the GPU mode
    (native.gpu_mode: GAMELAB_GPU, else the checkout's lab/GPU), and any other GAMELAB_* variable as set (NOT_KEYED
    aside). An unset GAMELAB_CHARACTER and one naming the thief, or GAMELAB_GPU unset and 0, are one key. (Absolute
    imports: lib/baseline.py, a copy of this file, runs it inside an older checkout's gamelab.)"""
    from gamelab import character, native
    environ = os.environ if environ is None else environ
    other = {k: v for k, v in sorted(environ.items())
             if k.startswith("GAMELAB_") and k not in NOT_KEYED and k not in RESOLVED}
    return {"character": character.active_id(environ), "fixed_stats": bool(environ.get(character.FIXED_STATS)),
            "gpu": native.gpu_mode(environ), **({"other": other} if other else {})}


def filler_env(env: dict, environ=None) -> dict:
    """The environment for the base's filler (base.py, in the base's checkout) whose runs' keys resolve to `env`
    (gamelab_env's, as `ab` keyed the base): the character and the GPU mode said outright, so a lab/GPU file or a
    character the candidate's process resolved some other way gives the base's key the same."""
    environ = dict(os.environ if environ is None else environ)
    environ["GAMELAB_CHARACTER"] = env["character"]
    environ["GAMELAB_GPU"] = {"off": "0", "start": "1", "ahead": "ahead"}[env["gpu"]]
    if env["fixed_stats"]:
        environ["GAMELAB_FIXED_STATS"] = "1"
    return environ


def environment() -> dict:
    """Python's version, the installed packages' (packages) and the GAMELAB_* environment as resolved (gamelab_env)."""
    return {"python": platform.python_version(), "packages": packages(), "gamelab": gamelab_env()}


def covered(p: dict, repo: Path = REPO) -> dict:
    """Everything the key covers, as meta.json shows it."""
    return {"format": FORMAT, "params": p, "data": data_stamp(repo), "env": environment()}


def key(p: dict, repo: Path = REPO) -> str:
    return hashlib.sha256(json.dumps(covered(p, repo), sort_keys=True).encode()).hexdigest()[:16]


class Cache:
    """The cached seeds of one run's arguments `p` on code `code` (code_id: a commit's): get, put, and which there are.
    `commit`: the commit running them, for meta.json."""

    def __init__(self, code: str, p: dict, repo: Path = REPO, root: Path | None = None, commit: str | None = None):
        self.code, self.params, self.repo, self.commit = code, p, repo, commit
        self.meta = covered(p, repo)
        self.key = hashlib.sha256(json.dumps(self.meta, sort_keys=True).encode()).hexdigest()[:16]
        self.folder = (root or ROOT) / "harness" / code / self.key

    def path(self, seed: int) -> Path:
        return self.folder / f"seed-{seed:03d}.json"

    def seeds(self) -> list[int]:
        return sorted(int(p.stem[5:]) for p in self.folder.glob("seed-*.json")) if self.folder.is_dir() else []

    def get(self, seed: int) -> dict | None:
        path = self.path(seed)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def put(self, seed: int, result: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        meta = self.folder / "meta.json"
        if not meta.exists():
            _write(meta, json.dumps({**self.meta, "code": self.code, "commit": self.commit}, indent=1,
                                    sort_keys=True))
        _write(self.path(seed), json.dumps(result))


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def rev(ref: str, repo: Path = REPO) -> str:
    """A ref's commit SHA (`master`, a short SHA), as the checkout's git has it; ValueError if there's none."""
    done = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if done.returncode != 0:
        raise ValueError(f"no commit {ref!r} ({done.stderr.strip()})")
    return done.stdout.strip()


def head(repo: Path = REPO) -> str | None:
    """The checkout's HEAD commit, or None."""
    done = _git(repo, "rev-parse", "--verify", "HEAD")
    return done.stdout.strip() if done.returncode == 0 else None


def plain(result: dict) -> dict:
    """A result as it reads back from its file (through JSON: lists for tuples, string keys)."""
    return json.loads(json.dumps(result))
