"""The baseline cache (gamelab.sim.baseline), the harness's runs through it (`run --cache`, `cached`, `compare` on a
cache's folder), and `ab`'s escalation."""

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from gamelab.sim import baseline, harness, scenarios

RUN = ("loop", 0, "crowd", 5.0, 0.0, "measured", None, 10002073, ("zero-walls", "near-misses", "walkers"), (), "choice")


def _key(args=RUN, fn=harness.run_one):
    return baseline.key(baseline.params(fn, args))


def test_the_key_covers_every_argument_of_a_run_but_the_seed():
    p = baseline.params(harness.run_one, RUN)
    assert set(p) == set(baseline.CORE) and "seed" not in p
    assert p["world"] == ["zero-walls", "near-misses", "walkers"] and p["minutes"] == 5.0
    other = {"policy": "loop-grind", "preset": "crowd-17", "minutes": 10.0, "think": 1.0, "costs": "coded",
             "ways": 3, "map_id": 10000012, "world": ("zero-walls",), "grounds": (23,), "attack": "skill",
             "think_clock": 1.0}
    names = list(baseline.inspect.signature(harness.run_one).parameters)
    assert set(other) == set(names) - {"seed"}  # (every argument of run_one tried)
    keys = {_key()}
    for name, value in other.items():
        args = list(RUN) + [0.0] * (len(names) - len(RUN))  # (a later switch at its default: think_clock)
        args[names.index(name)] = value
        keys.add(_key(tuple(args)))
    assert len(keys) == len(other) + 1  # each argument changes the key
    seeded = list(RUN)
    seeded[1] = 9
    assert _key(tuple(seeded)) == _key()  # ... but the seed: a file of its own
    same = list(RUN)
    same[3], same[8] = 5, ["zero-walls", "near-misses", "walkers"]  # 5 minutes, a list: the same run
    assert _key(tuple(same)) == _key()
    order = list(RUN)
    order[8] = ("near-misses", "zero-walls", "walkers")  # (the worlds in the order given: no guess they commute)
    assert _key(tuple(order)) != _key()


def test_a_switch_a_later_commit_adds_counts_only_when_its_not_at_its_default():
    def later(policy, seed, preset="game", minutes=20.0, think=0.0, costs="coded", ways=None, map_id=10000012,
              world=(), grounds=(), attack=None, ground_tie=None):
        pass
    assert _key(fn=later) == _key()  # off: the key of every run before it
    assert _key(RUN + ("closest",), fn=later) != _key()
    assert baseline.params(later, RUN + ("closest",))["ground_tie"] == "closest"


def test_the_key_covers_the_environment_as_it_resolves_and_the_game_data(monkeypatch, tmp_path):
    from gamelab import paths
    for name in [k for k in baseline.os.environ if k.startswith("GAMELAB_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("GAMELAB_FIXED_STATS", "1")  # (as the harness runs)
    monkeypatch.setattr(paths, "GPU", tmp_path / "GPU")  # (no lab/GPU file)
    before = _key()
    assert baseline.gamelab_env() == {"character": "thief", "fixed_stats": True, "gpu": "off"}
    for name, value in (("GAMELAB_BASELINES", "elsewhere"), ("GAMELAB_BASE_FILL", "fill.py"),  # where and who
                        ("GAMELAB_CHARACTER", "thief"), ("GAMELAB_GPU", "0")):  # ... and what they are by default
        monkeypatch.setenv(name, value)
        assert _key() == before, name
    for name, value in (("GAMELAB_GPU", "1"), ("GAMELAB_GPU", "ahead"), ("GAMELAB_CHARACTER", "magician"),
                        ("GAMELAB_SOMETHING_NEW", "1")):  # (one no code reads yet: kept, to be safe)
        monkeypatch.setenv(name, value)
        assert _key() != before, (name, value)
        monkeypatch.delenv(name)
    assert _key() == before
    (tmp_path / "GPU").write_text("ahead")  # the checkout's lab/GPU file, with GAMELAB_GPU unset ...
    assert baseline.gamelab_env()["gpu"] == "ahead" and _key() != before
    monkeypatch.setenv("GAMELAB_GPU", "0")  # ... which GAMELAB_GPU wins over
    assert _key() == before
    real = baseline.importlib.metadata.version
    monkeypatch.setattr(baseline.importlib.metadata, "version",  # OpenCV upgraded
                        lambda d: "9.9.9" if d == "opencv-python" else real(d))
    assert _key() != before
    monkeypatch.setattr(baseline.importlib.metadata, "version", real)
    assert _key() == before
    data = tmp_path / "gamedata" / "data" / "out" / "maps"
    data.mkdir(parents=True)
    (data / "1.json").write_text("{}")
    stamp = baseline.data_stamp(tmp_path)
    assert baseline.key(baseline.params(harness.run_one, RUN), tmp_path) != before
    (data / "1.json").write_text("{ }")  # extracted again
    assert baseline.data_stamp(tmp_path) != stamp


def test_the_bases_filler_gets_the_environment_its_key_resolved(monkeypatch, tmp_path):
    from gamelab import paths
    monkeypatch.setattr(paths, "GPU", tmp_path / "GPU")
    (tmp_path / "GPU").write_text("")  # the candidate's lab/GPU: the GPU on, at rf_start
    env = baseline.gamelab_env({"GAMELAB_FIXED_STATS": "1", "GAMELAB_BASE_FILL": "f.py", "GAMELAB_X": "y"})
    assert env == {"character": "thief", "fixed_stats": True, "gpu": "start", "other": {"GAMELAB_X": "y"}}
    filler = baseline.filler_env(env, {"PATH": "p", "GAMELAB_X": "y"})
    assert filler == {"PATH": "p", "GAMELAB_X": "y", "GAMELAB_CHARACTER": "thief", "GAMELAB_GPU": "1",
                      "GAMELAB_FIXED_STATS": "1"}
    (tmp_path / "GPU").unlink()  # the base's checkout has none
    assert baseline.gamelab_env(filler) == env


def test_a_run_imports_no_installed_package_the_key_hasnt():
    """The key has the versions of PACKAGES' distributions: a run of the standard command (Tree Dungeon, the feed's
    measured costs) importing any other installed package needs it added there."""
    if not scenarios.available(10002073):
        pytest.skip("needs Tree Dungeon's data")
    code = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "from gamelab.sim import harness\n"
        "harness.run_one('loop-grind', 1, 'crowd', 0.1, 0.0, 'measured', None, 10002073,\n"
        "                ('zero-walls', 'near-misses', 'walkers'), (), 'choice')\n"
        "new = {m.split('.')[0] for m in sys.modules} - {m.split('.')[0] for m in before}\n"
        "site = sorted(t for t in new if 'site-packages' in (getattr(sys.modules.get(t), '__file__', '') or ''))\n"
        "print(json.dumps(site))\n")
    env = dict(baseline.os.environ, PYTHONHASHSEED="0", GAMELAB_FIXED_STATS="1")
    done = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=str(harness.REPO), env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    site = json.loads(done.stdout.strip().splitlines()[-1])
    assert "cv2" in site and "numpy" in site  # (it saw them)
    assert set(site) <= set(baseline.PACKAGES), set(site) - set(baseline.PACKAGES)


def test_think_isnt_cached():
    assert baseline.uncacheable(baseline.params(harness.run_one, RUN)) is None
    thinking = RUN[:4] + (1.0,) + RUN[5:]
    assert "think" in baseline.uncacheable(baseline.params(harness.run_one, thinking))
    cache, why = harness.cache_of("loop", "crowd", 5.0, 1.0, code="f" * 40)
    assert cache is None and "think" in why
    cache, why = harness.cache_of("loop", "crowd", 5.0, think_clock=1.0, code="f" * 40)  # (--think-clock)
    assert cache is None and "think_clock" in why


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_the_codes_identity_is_its_commits_code_only_while_its_as_committed(tmp_path):
    repo = tmp_path / "repo"
    for rel in ("gamelab/a.py", "gamelab/README.md", "config.toml", "tests/test_a.py", "maplehelper/b.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x = 1\n")
    (repo / ".gitignore").write_text("gamelab/lab/\n")
    _git(tmp_path, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    sha = _git(repo, "rev-parse", "HEAD")
    code = baseline.code_id(sha, repo)
    assert baseline.identity(repo) == (code, "") and len(code) == 40
    (repo / "gamelab" / "README.md").write_text("more\n")  # Markdown: nothing runs it
    (repo / "tests" / "test_a.py").write_text("x = 2\n")  # tests don't matter
    (repo / "gamelab" / "lab").mkdir()
    (repo / "gamelab" / "lab" / "run.json").write_text("{}")  # ignored
    assert baseline.identity(repo) == (code, "")
    for rel, text in (("gamelab/a.py", "x = 2\n"), ("config.toml", "y = 1\n"), ("gamelab/new.py", "\n"),
                      ("maplehelper/b.py", "x = 3\n")):
        path = repo / rel
        old = path.read_text() if path.exists() else None
        path.write_text(text)
        got, why = baseline.identity(repo)
        assert got is None and rel in why
        path.write_text(old) if old is not None else path.unlink()
    assert baseline.identity(repo) == (code, "")
    _git(repo, "add", "gamelab/a.py")
    assert baseline.identity(repo)[0] == code  # (as committed again: the index the same as HEAD)

    def commit(*rels: str) -> str:
        _git(repo, "add", *rels)
        _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
        return _git(repo, "rev-parse", "HEAD")
    readme_only = commit("gamelab/README.md", "tests/test_a.py")  # a README's and a test's commit: the same code
    assert readme_only != sha and baseline.code_id(readme_only, repo) == code == baseline.identity(repo)[0]
    (repo / "gamelab" / "a.py").write_text("x = 5\n")
    changed = commit("gamelab/a.py")
    assert baseline.code_id(changed, repo) != code == baseline.code_id(sha, repo)


def test_a_cache_keeps_a_seed_a_file_and_a_folder_of_them_loads_as_a_run(tmp_path):
    p = baseline.params(harness.run_one, RUN)
    cache = baseline.Cache("a" * 40, p, root=tmp_path, commit="e" * 40)
    assert cache.seeds() == [] and cache.get(1) is None
    cache.put(2, {"seed": 2, "exp_per_min": 1.5, "t": (1, 2)})
    cache.put(1, {"seed": 1, "exp_per_min": 2.5})
    assert cache.seeds() == [1, 2] and cache.get(2) == {"seed": 2, "exp_per_min": 1.5, "t": [1, 2]}
    assert cache.folder == tmp_path / "harness" / ("a" * 40) / cache.key
    meta = json.loads((cache.folder / "meta.json").read_text())
    assert meta["params"] == p and meta["code"] == "a" * 40 and meta["commit"] == "e" * 40
    assert meta["format"] == baseline.FORMAT
    assert [r["seed"] for r in harness.load(str(cache.folder))] == [1, 2]
    assert not list(cache.folder.glob("*.tmp"))


def test_a_cached_run_is_the_same_as_an_uncached_one(monkeypatch, tmp_path):
    if not scenarios.available(harness.TREE):
        pytest.skip("needs Tree Dungeon's data")
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "identity", lambda repo=None: ("c" * 40, ""))
    args = ("game", 0.25, 1, 0.0, "coded", None, harness.TREE, (), (), None)
    plain = harness.run_seeds("tour", [1, 2], *args)
    said = []
    first = harness.cached_run("tour", [1, 2], *args, say=said.append)  # run, and kept
    assert "0 of 2 seed(s) read" in said[0] and "2 to run" in said[0]
    monkeypatch.setattr(harness, "run_seeds", lambda *a, **k: pytest.fail("a cached seed was run again"))
    again = harness.cached_run("tour", [2, 1], *args, say=said.append)  # read, in the order asked
    assert "2 of 2 seed(s) read" in said[1]
    assert json.dumps(first) == json.dumps(baseline.plain(plain)) == json.dumps(again[::-1])


def test_a_checkout_with_changes_runs_everything_and_keeps_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "identity", lambda repo=None: (None, "the checkout's code isn't as committed (x)"))
    monkeypatch.setattr(harness, "run_seeds", lambda policy, seeds, *a, **k: [{"seed": s, "t": (s,)} for s in seeds])
    said = []
    got = harness.cached_run("tour", [1, 2], "game", 1.0, say=said.append)
    assert got == [{"seed": 1, "t": [1]}, {"seed": 2, "t": [2]}] and "not used" in said[0]
    assert not list(tmp_path.rglob("*.json"))


def test_seeds_run_on_a_pool_come_back_in_the_order_asked(monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "run_one", lambda *a: {"seed": a[1]})
    got = harness.run_seeds("tour", [3, 1, 2], "game", 1.0, done=lambda k, r: calls.append(k))
    assert [r["seed"] for r in got] == [3, 1, 2] and calls == [3, 1, 2]


# --- ab -------------------------------------------------------------------------------------------------

def _runs(exp: list[float], touches: list[int] | None = None, **more) -> list[dict]:
    touches = touches or [10] * len(exp)
    return [{"seed": k + 1, "map": 10002073, "exp_per_min": e, "touches": t, "died_at": None, "fell_out_at": None,
             "left_map_at": None, "safety_stops": 0, "darts_stops": 0, "drops_share": 0.5, "hp_lost": 10.0,
             "backstops": {},
             **{name: v[k] for name, v in more.items()}}
            for k, (e, t) in enumerate(zip(exp, touches))]


def test_a_difference_is_paired_seed_by_seed_with_its_standard_error():
    base = _runs([100.0, 110.0, 90.0, 100.0])
    cand = _runs([102.0, 113.0, 91.0, 102.0])
    row = harness.difference(base, cand, "exp_per_min")
    assert row["n"] == 4 and row["diff"] == pytest.approx(2.0) and row["base"] == pytest.approx(100.0)
    assert row["se"] == pytest.approx(0.8165 / 2, abs=1e-4) and row["verdict"] == "beyond 2 se"
    assert harness.difference(base, base, "touches")["verdict"] == "zero"
    died = [dict(r, died_at=60.0 if r["seed"] == 2 else None) for r in base]
    assert harness.difference(base, base, "deaths")["verdict"] == "zero"
    assert harness.difference(base, died, "deaths")["diff"] == pytest.approx(0.25)
    noisy = _runs([100.0, 120.0, 80.0, 101.0])
    assert harness.difference(base, noisy, "exp_per_min")["verdict"] == "within 2 se"
    one = _runs([100.0, 100.0, 100.0, 100.0], mp_per_kill=[None, None, 5.0, None])
    other = _runs([100.0, 100.0, 100.0, 100.0], mp_per_kill=[None, None, 7.0, None])
    row = harness.difference(one, other, "mp_per_kill")  # a value on both sides on one seed: no se to tell it by
    assert row["n"] == 1 and row["verdict"] == "too few"
    assert harness.difference(one, one, "mp_per_kill")["verdict"] == "zero"  # (the same, None and all)
    gone = _runs([100.0] * 4, mp_per_kill=[None, 3.0, 5.0, None])  # seed 2 a kill here only
    assert harness.difference(one, gone, "mp_per_kill")["verdict"] == "too few"
    assert harness.difference(base[:0], base[:0], "touches")["verdict"] == "too few"  # (no seeds at all)


def test_differences_that_offset_are_not_zero():
    """Seeds that differ with a mean difference of 0 (a death moved from seed 7 to seed 3, +4 touches on one and -4 on
    another, +30 EXP a minute and -30) are within 2 se, not 'the same to the digit': more seeds are taken."""
    base = _runs([100.0] * 12, [10] * 12)
    cand = [dict(r) for r in base]
    base[6]["died_at"], cand[2]["died_at"] = 120.0, 200.0
    cand[0]["touches"], cand[1]["touches"] = 14, 6
    cand[4]["exp_per_min"], cand[5]["exp_per_min"] = 130.0, 70.0
    rows = [harness.difference(base, cand, m) for m in ("deaths", "touches", "exp_per_min", "falls", "stops")]
    assert [r["diff"] for r in rows] == [0.0] * 5
    assert [r["verdict"] for r in rows] == ["within 2 se", "within 2 se", "within 2 se", "zero", "zero"]
    assert rows[0]["se"] > 0 and rows[1]["se"] > 0
    more, why = harness.ab_next(rows, 12)
    assert more == 24 and "deaths, touches, exp_per_min" in why
    assert harness.metric({"landings": {"trip": {"near": 3}}}, "landings.trip.near") == 3
    assert harness.metric({"landings": {}}, "landings.trip.near") == 0  # (a count that's absent)
    assert harness.metric({"mp_per_kill": None}, "mp_per_kill") is None


def test_ab_adds_seeds_only_while_a_headline_difference_is_within_2_se_and_not_zero():
    clear = [{"metric": "exp_per_min", "verdict": "beyond 2 se"}, {"metric": "deaths", "verdict": "zero"}]
    assert harness.ab_next(clear, 12) == (None, "every headline difference is beyond 2 se or zero at 12 seeds")
    open_ = clear + [{"metric": "touches", "verdict": "within 2 se"}]
    more, why = harness.ab_next(open_, 12)
    assert more == 24 and "touches" in why
    assert harness.ab_next(open_, 24)[0] == 48
    more, why = harness.ab_next(open_, 48)
    assert more is None and "the most" in why and "still unsettled: within 2 se and not zero: touches" in why
    assert harness.ab_next(open_, 24, (12, 24))[0] is None  # (--max-seeds 24)
    few = clear + [{"metric": "mp_per_kill", "verdict": "too few"}]  # too few to tell: more seeds, said so
    more, why = harness.ab_next(few, 12)
    assert more == 24 and why == "at 12 seeds too few seeds with a value on both sides: mp_per_kill"
    more, why = harness.ab_next(few + open_[2:], 48)
    assert more is None and why.endswith("still unsettled: within 2 se and not zero: touches; too few seeds with a "
                                         "value on both sides: mp_per_kill")


def _ab_args(**kw) -> Namespace:
    a = dict(policy="loop", seeds=12, minutes=5.0, preset="crowd", jobs=1, think=0.0, think_clock=0.0, save="abtest",
             ways=None,
             costs="measured", map=10002073, world=["walkers"], grounds=(), attack="choice", base="master",
             metric=["hurt"], max_seeds=48, fill=None)
    a.update(kw)
    return Namespace(**a)


def test_ab_reads_the_base_from_its_cache_and_says_the_seeds_it_used_and_why_it_stopped(monkeypatch, tmp_path,
                                                                                           capsys):
    import random
    rng = random.Random(3)
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "rev", lambda ref, repo=None: "b" * 40)
    monkeypatch.setattr(baseline, "code_id", lambda commit, repo=None: commit)
    monkeypatch.setattr(harness, "FOLDER", tmp_path / "runs")
    monkeypatch.setattr(baseline, "identity", lambda repo=None: (None, "changes"))
    base = _runs([100.0 + rng.gauss(0, 10) for _ in range(48)], hurt=[5] * 48)
    cache, _ = harness.cache_of("loop", *harness._run_args(_ab_args()), code="b" * 40)
    for r in base:
        cache.put(r["seed"], r)
    asked = []

    def cand_run(policy, seeds, *a, say=print, **k):  # the candidate: a touch more EXP, noisy
        asked.append(list(seeds))
        return [dict(base[s - 1], exp_per_min=base[s - 1]["exp_per_min"] + 1 + rng.gauss(0, 10)) for s in seeds]
    monkeypatch.setattr(harness, "cached_run", cand_run)
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: pytest.fail("the base's seeds are all cached"))
    assert harness.ab(_ab_args()) == 0
    out = capsys.readouterr().out
    assert asked == [list(range(1, 13)), list(range(13, 25)), list(range(25, 49))]
    assert "-> 24 seeds: at 12 seeds within 2 se and not zero: exp_per_min" in out
    assert "stopped at 48 seeds (1-48): 48 seeds, the most (--max-seeds); still unsettled: within 2 se and not zero: " \
           "exp_per_min" in out
    assert "hurt" in out  # (the targeted metric beside the headline ones)
    assert len(harness.load("abtest")) == 48 and harness.load("abtest-base") == base

    asked.clear()
    monkeypatch.setattr(harness, "cached_run", lambda policy, seeds, *a, say=print, **k:
                        asked.append(list(seeds)) or [dict(base[s - 1]) for s in seeds])  # the same: all zero
    assert harness.ab(_ab_args(save="same")) == 0
    out = capsys.readouterr().out
    assert asked == [list(range(1, 13))] and "stopped at 12 seeds (1-12): every headline difference is beyond 2 " \
                                                "se or zero at 12 seeds" in out


def test_ab_has_the_filler_run_the_bases_missing_seeds(monkeypatch, tmp_path, capsys):
    from gamelab import paths
    monkeypatch.setattr(paths, "GPU", tmp_path / "GPU")  # (no lab/GPU file here)
    monkeypatch.delenv("GAMELAB_GPU", raising=False)
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "rev", lambda ref, repo=None: "b" * 40)
    monkeypatch.setattr(baseline, "code_id", lambda commit, repo=None: commit)
    monkeypatch.setattr(harness, "FOLDER", tmp_path / "runs")
    monkeypatch.setattr(baseline, "identity", lambda repo=None: (None, "changes"))
    base = _runs([100.0] * 12, hurt=[1] * 12)
    cache, _ = harness.cache_of("loop", *harness._run_args(_ab_args()), code="b" * 40)
    filler = tmp_path / "fill.py"
    filler.write_text("")
    commands = []

    def fill(command, **k):
        commands.append(command)
        env = k["env"]  # the environment the base's key resolved, said outright
        assert env["GAMELAB_CHARACTER"] == "thief" and env["GAMELAB_GPU"] == "0"
        assert baseline.gamelab_env(env) == cache.meta["env"]["gamelab"]
        for r in base:
            cache.put(r["seed"], r)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(harness.subprocess, "run", fill)
    monkeypatch.setattr(harness, "cached_run", lambda policy, seeds, *a, say=print, **k: [base[s - 1] for s in seeds])
    assert harness.ab(_ab_args(fill=str(filler), jobs=2)) == 0
    assert commands == [[sys.executable, str(filler), "--ref", "b" * 40, "--", "run", "loop", "--preset", "crowd",
                         "--minutes", "5.0", "--costs", "measured", "--map", "10002073", "--world", "walkers",
                         "--attack", "choice", "--seeds", "12", "--jobs", "2"]]
    assert "base: seed(s) 1-12 not cached" in capsys.readouterr().out


def test_ab_refuses_a_metric_no_seed_has(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "rev", lambda ref, repo=None: "b" * 40)
    monkeypatch.setattr(baseline, "code_id", lambda commit, repo=None: commit)
    monkeypatch.setattr(harness, "FOLDER", tmp_path / "runs")
    monkeypatch.setattr(baseline, "identity", lambda repo=None: (None, "changes"))
    base = _runs([100.0] * 48, landings=[{"trip": {"near": 1}}] * 48)
    cache, _ = harness.cache_of("loop", *harness._run_args(_ab_args()), code="b" * 40)
    for r in base:
        cache.put(r["seed"], r)
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: pytest.fail("the base's seeds are all cached"))
    asked = []
    new_key = [dict(r, fresh={"n": r["seed"] % 2}) for r in base]  # a metric the candidate's change adds
    monkeypatch.setattr(harness, "cached_run", lambda policy, seeds, *a, say=print, **k:
                        asked.append(list(seeds)) or [new_key[s - 1] for s in seeds])
    assert harness.ab(_ab_args(metric=["landings.trip.neer"])) == 2
    out = capsys.readouterr().out
    assert "--metric landings.trip.neer: in none of the base's results" in out
    assert "--metric landings.trip.neer: no seed's result has it, on either side (a typo?); did you mean " \
           "landings.trip.near" in out
    assert asked == [list(range(1, 13))] and "seeds 1-12" not in out  # (refused before any table)
    assert harness.ab(_ab_args(metric=["hurts"])) == 2 and "--metric hurts: no seed's result" in capsys.readouterr().out
    assert harness.ab(_ab_args(metric=["fresh.n", "landings.trip.near"], save="fresh")) == 0
    out = capsys.readouterr().out
    assert "--metric fresh.n: in none of the base's results" in out and "fresh.n" in out.split("seeds 1-12:")[1]


def test_a_spot_check_runs_a_saved_runs_seeds_again_with_their_own_arguments(monkeypatch):
    saved = [{"seed": k, "policy": "loop", "preset": "crowd", "minutes": 5.0, "think": 0.0, "costs": "measured",
              "ways": 1, "map": 10002073, "world": ["walkers"], "attack": "choice", "exp_per_min": 100.0 + k,
              "touches": 10, "nav": {"misses": 2}} for k in (1, 2, 3)]
    calls = []

    def again(policy, seeds, preset, minutes, jobs, think, costs, ways, map_id, world, grounds, attack, think_clock):
        calls.append((policy, list(seeds), preset, minutes, think, costs, ways, map_id, world, grounds, attack,
                      think_clock))
        return [dict(saved[k - 1], nav={"misses": 2}, t=(1,)) if k != 3 else dict(saved[2], touches=11, t=(1,))
                for k in seeds]
    monkeypatch.setattr(harness, "run_seeds", again)
    said = []
    saved = [dict(r, t=[1]) for r in saved]
    assert harness.spot(saved, [1, 2], say=said.append)
    assert calls == [("loop", [1, 2], "crowd", 5.0, 0.0, "measured", 1, 10002073, ("walkers",), (), "choice", 0.0)]
    assert said[0].startswith("seed 1: the same to the digit")
    said.clear()
    assert not harness.spot(saved, [3], say=said.append)
    assert said[0].startswith("seed 3: DIFFERS in touches") and "saved 10, now 11" in said[0]
    assert not harness.spot(saved, [7], say=said.append) and "no seed 7" in said[-1]


def test_cached_reads_a_commits_seeds_and_says_how_to_fill_the_missing_ones(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "rev", lambda ref, repo=None: "d" * 40)
    monkeypatch.setattr(baseline, "code_id", lambda commit, repo=None: commit)
    monkeypatch.setattr(harness, "FOLDER", tmp_path / "runs")
    args = _ab_args(seeds=3, sha="master", save=None)
    cache, _ = harness.cache_of("loop", *harness._run_args(args), code="d" * 40)
    for r in _runs([100.0, 101.0]):
        cache.put(r["seed"], r)
    assert harness._cached(args) == 1
    out = capsys.readouterr().out
    assert "2 of 3 seed(s) cached" in out and "missing: seed(s) 3" in out and "--ref " + "d" * 40 in out
    assert harness._cached(_ab_args(seeds=2, sha="master", save=None)) == 0
    assert [r["seed"] for r in harness.load(f"{harness._name(args)}@ddddddd")] == [1, 2]
