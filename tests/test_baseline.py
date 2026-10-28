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
             "ways": 3, "map_id": 10000012, "world": ("zero-walls",), "grounds": (23,), "attack": "skill"}
    names = list(baseline.inspect.signature(harness.run_one).parameters)
    assert set(other) == set(names) - {"seed"}  # (every argument of run_one tried)
    keys = {_key()}
    for name, value in other.items():
        args = list(RUN)
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


def test_the_key_covers_the_environment_and_the_game_data(monkeypatch, tmp_path):
    before = _key()
    monkeypatch.setenv("GAMELAB_BASELINES", "elsewhere")  # where, not what
    assert _key() == before
    monkeypatch.setenv("GAMELAB_GPU", "1")
    assert _key() != before
    monkeypatch.delenv("GAMELAB_GPU")
    data = tmp_path / "gamedata" / "data" / "out" / "maps"
    data.mkdir(parents=True)
    (data / "1.json").write_text("{}")
    stamp = baseline.data_stamp(tmp_path)
    assert baseline.key(baseline.params(harness.run_one, RUN), tmp_path) != before
    (data / "1.json").write_text("{ }")  # extracted again
    assert baseline.data_stamp(tmp_path) != stamp


def test_think_isnt_cached():
    assert baseline.uncacheable(baseline.params(harness.run_one, RUN)) is None
    thinking = RUN[:4] + (1.0,) + RUN[5:]
    assert "think" in baseline.uncacheable(baseline.params(harness.run_one, thinking))
    cache, why = harness.cache_of("loop", "crowd", 5.0, 1.0, sha="f" * 40)
    assert cache is None and "think" in why


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_the_codes_identity_is_its_commit_only_while_its_as_committed(tmp_path):
    repo = tmp_path / "repo"
    for rel in ("gamelab/a.py", "gamelab/README.md", "config.toml", "tests/test_a.py", "maplehelper/b.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x = 1\n")
    (repo / ".gitignore").write_text("gamelab/lab/\n")
    _git(tmp_path, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    sha = _git(repo, "rev-parse", "HEAD")
    assert baseline.identity(repo) == (sha, "")
    (repo / "gamelab" / "README.md").write_text("more\n")  # Markdown: nothing runs it
    (repo / "tests" / "test_a.py").write_text("x = 2\n")  # tests don't matter
    (repo / "gamelab" / "lab").mkdir()
    (repo / "gamelab" / "lab" / "run.json").write_text("{}")  # ignored
    assert baseline.identity(repo) == (sha, "")
    for rel, text in (("gamelab/a.py", "x = 2\n"), ("config.toml", "y = 1\n"), ("gamelab/new.py", "\n"),
                      ("maplehelper/b.py", "x = 3\n")):
        path = repo / rel
        old = path.read_text() if path.exists() else None
        path.write_text(text)
        got, why = baseline.identity(repo)
        assert got is None and rel in why
        path.write_text(old) if old is not None else path.unlink()
    assert baseline.identity(repo) == (sha, "")
    _git(repo, "add", "gamelab/a.py")
    assert baseline.identity(repo)[0] == sha  # (as committed again: the index the same as HEAD)


def test_a_cache_keeps_a_seed_a_file_and_a_folder_of_them_loads_as_a_run(tmp_path):
    p = baseline.params(harness.run_one, RUN)
    cache = baseline.Cache("a" * 40, p, root=tmp_path)
    assert cache.seeds() == [] and cache.get(1) is None
    cache.put(2, {"seed": 2, "exp_per_min": 1.5, "t": (1, 2)})
    cache.put(1, {"seed": 1, "exp_per_min": 2.5})
    assert cache.seeds() == [1, 2] and cache.get(2) == {"seed": 2, "exp_per_min": 1.5, "t": [1, 2]}
    assert cache.folder == tmp_path / "harness" / ("a" * 40) / cache.key
    meta = json.loads((cache.folder / "meta.json").read_text())
    assert meta["params"] == p and meta["sha"] == "a" * 40 and meta["format"] == baseline.FORMAT
    assert [r["seed"] for r in harness.load(str(cache.folder))] == [1, 2]
    assert not list(cache.folder.glob("*.tmp"))


def test_a_cached_run_is_the_same_as_an_uncached_one(monkeypatch, tmp_path):
    if not scenarios.available():
        pytest.skip("needs the real map's data")
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "identity", lambda repo=None: ("c" * 40, ""))
    args = ("game", 0.25, 1, 0.0, "coded", None, harness.FIRST_MAP, (), (), None)
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
    assert more is None and "the most" in why and "touches" in why
    assert harness.ab_next(open_, 24, (12, 24))[0] is None  # (--max-seeds 24)


def _ab_args(**kw) -> Namespace:
    a = dict(policy="loop", seeds=12, minutes=5.0, preset="crowd", jobs=1, think=0.0, save="abtest", ways=None,
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
    monkeypatch.setattr(harness, "FOLDER", tmp_path / "runs")
    monkeypatch.setattr(baseline, "identity", lambda repo=None: (None, "changes"))
    base = _runs([100.0 + rng.gauss(0, 10) for _ in range(48)], hurt=[5] * 48)
    cache, _ = harness.cache_of("loop", *harness._run_args(_ab_args()), sha="b" * 40)
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
    assert "-> 24 seeds: within 2 se and not zero at 12 seeds: exp_per_min" in out
    assert "stopped at 48 seeds (1-48): 48 seeds, the most (--max-seeds); still within 2 se: exp_per_min" in out
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
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "rev", lambda ref, repo=None: "b" * 40)
    monkeypatch.setattr(harness, "FOLDER", tmp_path / "runs")
    monkeypatch.setattr(baseline, "identity", lambda repo=None: (None, "changes"))
    base = _runs([100.0] * 12)
    cache, _ = harness.cache_of("loop", *harness._run_args(_ab_args()), sha="b" * 40)
    filler = tmp_path / "fill.py"
    filler.write_text("")
    commands = []

    def fill(command, **k):
        commands.append(command)
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
