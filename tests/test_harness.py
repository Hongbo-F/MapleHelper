"""The simulator's realism knobs (all off by default) and the results harness that compares hunting
policies on the same seeds (gamelab.sim.harness). Synthetic maps need no game data; the real-map test is
skipped without it."""

import dataclasses
import random
from collections import Counter

import pytest

from gamelab.behavior.hunter import Hunter
from gamelab.localization import mapdata
from gamelab.localization.pose import Localizer
from gamelab.sim import fakes, harness, scenarios
from gamelab.sim import world as world_mod
from gamelab.sim.fakes import Clock, FakeEyes, SimTimeout
from gamelab.sim.world import DROP_LIFE, Noise, Realism, SimPet, World

from maplehelper.keys import parse as K
from test_sim import KEYS, plat, still, tap, tree


@pytest.fixture(scope="module")
def model():
    return mapdata.parse(1, tree(), "Test Map", {}, {"2": "Snail"}, {"2": 2})


def world(model, pid, x, **kw):
    kw.setdefault("monsters", None)
    return World(model, KEYS, (pid, x), **kw)


def test_a_hit_monster_hides_for_a_while_and_a_look_can_miss_one(model):
    b = plat(model, 360, 300)
    clock = Clock()
    w = still(World(model, KEYS, (b.id, 240.0), monsters=[("Pig", b.id, 380.0)], origin=clock.now,
                    noise=Noise(hit_hidden=(1.0, 3.0))))
    eyes = FakeEyes(w, clock, model)
    h = Hunter(eyes, KEYS, model, log=lambda m: None)
    assert len(h.look().monsters) == 1
    w.monsters[0].hit_at = w.t  # a star just hit it
    assert h.look().monsters == []
    clock.advance_to(clock.now + 3.1)
    assert len(h.look().monsters) == 1  # back after 1-3 s
    w = World(model, KEYS, (b.id, 240.0), monsters=[("Pig", b.id, 380.0)], origin=clock.now,
              noise=Noise(detect_miss=1.0))
    assert Hunter(FakeEyes(w, clock, model), KEYS, model, log=lambda m: None).look().monsters == []


def test_near_her_on_her_platform_a_look_finds_a_monster_as_seldom_as_on_2026_10_25(model):
    """`--world near-misses-1025` (`near-misses` until 2026-11-03; Noise.near_seen, world.NEAR_SEEN_1025,
    OFF_SEEN_1025; the live hunt of 2026-10-25): a monster on her platform within 25 units of her is never found, within
    50 found by 7% of looks, within 100 by 34%, within 150 by 55%, farther by 60%; one on another platform by 70%
    (Noise.off_seen; without it, as the preset's detect_miss has it); a draw a look, no spells, over `near-misses`'
    whatever the order (harness.LAST_WORLDS)."""
    c = plat(model, 500, 400)
    clock = Clock()

    def found(x: float, pid: int = c.id, looks: int = 400, off: float | None = None) -> float:
        w = still(World(model, KEYS, (c.id, 100.0), monsters=[("Snail", pid, x)], origin=clock.now,
                        noise=Noise(near_seen=world_mod.NEAR_SEEN_1025, off_seen=off)))
        w.invulnerable_ms = 10 ** 9
        eyes = FakeEyes(w, clock, model)
        h = Hunter(eyes, KEYS, model, log=lambda m: None)
        n = 0
        for _ in range(looks):
            n += bool(h.look(wide=True).monsters)
            clock.advance_to(clock.now + 0.05)
        return n / looks
    assert found(115.0) == 0.0  # 15 units off: under her
    assert found(140.0) == pytest.approx(0.07, abs=0.04)
    assert found(180.0) == pytest.approx(0.34, abs=0.07)
    assert found(230.0) == pytest.approx(0.55, abs=0.07)
    assert found(400.0) == pytest.approx(0.6, abs=0.07)  # farther along her platform
    b = plat(model, 360, 300)
    assert found(300.0, b.id) == 1.0  # on another platform: as detect_miss has it (none here) ...
    assert found(300.0, b.id, off=world_mod.OFF_SEEN_1025) == pytest.approx(0.7, abs=0.07)  # ... or off_seen
    assert harness.WORLDS["near-misses-1025"] == {"near_seen": world_mod.NEAR_SEEN_1025,
                                                  "off_seen": world_mod.OFF_SEEN_1025, "near_spell": (),
                                                  "off_spell": 0.0}
    assert {"near_seen", "off_seen", "near_spell", "off_spell"} <= set(harness.NOISE_FIELDS)
    for world in (("near-misses-1025",), ("near-misses", "near-misses-1025"), ("near-misses-1025", "near-misses")):
        noise = harness.scenario("loop", 1, "crowd", 1.0, map_id=harness.TREE, world=world).noise
        assert (noise.near_seen, noise.off_seen, noise.near_spell, noise.off_spell) == \
            (world_mod.NEAR_SEEN_1025, world_mod.OFF_SEEN_1025, (), 0.0)


def test_near_misses_finds_a_monster_near_her_as_the_perception_of_2026_11_02(model):
    """`--world near-misses` from 2026-11-03 (and `near-misses-1102`, its name in the touch study; world.NEAR_SEEN,
    OFF_SEEN, NEAR_SPELL, OFF_SPELL: NEAR_SEEN_1102 ..., the replay kit's dumps of 92b09c3's perception on R1-R3 and
    BugBaby's frame sets): within 25 units of her 39% of looks find it, within 50 63%, within 100 80%, within 150 83%,
    farther 93%, on another platform 91%; each miss in a run, as live's (NEAR_SPELL_1102, OFF_SPELL_1102: a draw a
    spell, 0.33 s within 25 of her), where a draw a look finds one after a miss as often as after a find; the two
    worlds one, whatever the order (harness.LAST_WORLDS); the scenarios' detector as live (scenarios._NEAR) the
    same."""
    c = plat(model, 500, 400)
    clock = Clock()

    def looks(x: float, pid: int = c.id, n: int = 400, spells: bool = False) -> list[bool]:
        w = still(World(model, KEYS, (c.id, 100.0), monsters=[("Snail", pid, x)], origin=clock.now,
                        noise=Noise(near_seen=world_mod.NEAR_SEEN_1102, off_seen=world_mod.OFF_SEEN_1102,
                                    **({"near_spell": world_mod.NEAR_SPELL_1102, "off_spell": world_mod.OFF_SPELL_1102}
                                       if spells else {}))))
        w.invulnerable_ms = 10 ** 9
        h = Hunter(FakeEyes(w, clock, model), KEYS, model, log=lambda m: None)
        out = []
        for _ in range(n):
            out.append(bool(h.look(wide=True).monsters))
            clock.advance_to(clock.now + 0.05)
        return out

    def after(seen: list[bool], was: bool) -> float:
        pairs = [b for a, b in zip(seen, seen[1:]) if a == was]
        return sum(pairs) / len(pairs)
    for x, share in ((115.0, 0.39), (140.0, 0.63), (180.0, 0.80), (230.0, 0.83), (400.0, 0.93)):
        assert sum(looks(x)) / 400 == pytest.approx(share, abs=0.07)  # (a draw a look: the share alone)
    other = plat(model, 360, 300).id
    assert sum(looks(300.0, other)) / 400 == pytest.approx(0.91, abs=0.07)
    runs, flat = looks(180.0, n=1200, spells=True), looks(180.0, n=1200)  # (50-100 off: a spell 0.93 s; a look here
    assert sum(runs) / 1200 == pytest.approx(0.80, abs=0.07)                # 0.4 s, its own cost and the 0.05)
    assert after(runs, True) > 0.85 and after(runs, False) < 0.5
    assert after(flat, True) == pytest.approx(0.80, abs=0.07) and after(flat, False) == pytest.approx(0.80, abs=0.1)
    live = {"near_seen": world_mod.NEAR_SEEN_1102, "off_seen": world_mod.OFF_SEEN_1102,
            "near_spell": world_mod.NEAR_SPELL_1102, "off_spell": world_mod.OFF_SPELL_1102}
    assert harness.WORLDS["near-misses"] == harness.WORLDS["near-misses-1102"] == live
    assert (world_mod.NEAR_SEEN, world_mod.OFF_SEEN, world_mod.NEAR_SPELL, world_mod.OFF_SPELL) == tuple(live.values())
    for world in (("near-misses",), ("near-misses-1102",), ("near-misses", "near-misses-1102"),
                  ("near-misses-1102", "near-misses")):
        noise = harness.scenario("loop", 1, "crowd", 1.0, map_id=harness.TREE, world=world).noise
        assert (noise.near_seen, noise.off_seen, noise.near_spell, noise.off_spell) == tuple(live.values())
    near = scenarios._NEAR
    assert (near.near_seen, near.off_seen, near.near_spell, near.off_spell) == tuple(live.values())


def test_her_invulnerability_is_the_sources_1_5_s_in_every_world(model, monkeypatch):
    """Every world's invulnerability after a touch from 2026-11-03 (world.INVULNERABLE, INVULNERABLE_LIVE: `walkers`'
    too; INVULNERABLE_SOURCES, `--world invulnerable-1.5` the same): a monster still on her touches her again 1.5 s
    after a touch, or at 1.45 with `--world invulnerable-1.45` (INVULNERABLE_LEAST_GAP, every world's from 2026-10-28
    until then), whatever the order it and `walkers` are given in (harness.LAST_WORLDS); the stack's own figure
    (perception.touches.INVULNERABLE_S) 1.45, unchanged."""
    from gamelab.perception import touches
    monkeypatch.setattr(world_mod, "EVADE", 1.0)  # (every touch a MISS: no knock, the pig stays on her)
    c = plat(model, 500, 400)
    for inv in (world_mod.INVULNERABLE_LEAST_GAP, world_mod.Realism().invulnerable):
        w = still(world(model, c.id, 400.0, monsters=[("Pig", c.id, 410.0)], realism=Realism(invulnerable=inv)))
        w.advance(0.05)
        assert w.touches == 1 and w.evaded == 1
        first = w.touch_log[0][0]
        w.advance(first + inv - 0.02)  # (advance: to that time)
        assert w.touches == 1
        w.advance(first + inv + 0.02)
        assert w.touches == 2 and w.touch_log[1][0] - first == pytest.approx(inv, abs=0.006)
    assert world_mod.INVULNERABLE == world_mod.INVULNERABLE_LIVE == world_mod.INVULNERABLE_SOURCES == 1.5
    assert harness.WORLDS["invulnerable-1.5"] == {"invulnerable": 1.5}
    assert harness.WORLDS["invulnerable-1.45"] == {"invulnerable": 1.45}
    for inv, name in ((1.5, "invulnerable-1.5"), (1.45, "invulnerable-1.45")):
        for got in (("walkers", name), (name, "walkers")):
            sc = harness.scenario("loop", 1, "crowd", 1.0, map_id=harness.TREE, world=("zero-walls", *got))
            assert sc.realism.invulnerable == inv and sc.realism.chase
    for got in ((), ("walkers",)):
        assert harness.scenario("loop", 1, "crowd", 1.0, map_id=harness.TREE, world=got).realism.invulnerable == 1.5
    assert touches.INVULNERABLE_S == 1.45


def test_a_monster_back_at_its_spawn_point_is_found_once_its_fade_reaches_its_drawn_alpha(model):
    """`--world spawn-fade` and `spawn-fade-wide` (Noise.spawn_found: world.SPAWN_FOUND, SPAWN_FOUND_WIDE, measured on
    Tree Dungeon's clips): a monster back at its spawn point fades in over SPAWN_FADE_S, its alpha in what the looks
    see (Mob.fading) while it does, and isn't found before its alpha reaches the one drawn for it, the tables' medians
    0.098 (the spawn watch on) and 0.647 (the whole screen alone); a monster there from the start, or the noise off,
    is drawn solid and found from the start, as before."""
    for table, median in ((world_mod.SPAWN_FOUND, 0.098), (world_mod.SPAWN_FOUND_WIDE, 0.647)):
        draws = sorted(Noise.found_at(table, (k + 0.5) / 1000) for k in range(1000))
        assert draws[500] == pytest.approx(median, abs=0.01) and draws[0] >= table[0][0] and draws[-1] <= 1.0
    b = plat(model, 360, 300)
    clock = Clock()

    def first_found(noise: Noise):
        w = still(World(model, KEYS, (b.id, 100.0), monsters=[("Snail", b.id, 300.0)], origin=clock.now, noise=noise))
        w.invulnerable_ms = 10 ** 9
        h = Hunter(FakeEyes(w, clock, model), KEYS, model, log=lambda m: None)
        w.monsters[0].born = w.t   # (back at its spawn point now)
        for k in range(20):
            got = h.look(wide=True).monsters
            if got:
                return k, got[0].fading
            clock.advance_to(clock.now + 0.05)
        return None, None
    k, fading = first_found(Noise(spawn_found=world_mod.SPAWN_FOUND_WIDE))
    assert k >= 1 and 0.2 <= fading < 1.0   # (the first looks, too early: unfound)
    k0, fading0 = first_found(Noise())
    assert (k0, fading0) == (0, None)
    assert harness.WORLDS["spawn-fade"] == {"spawn_found": world_mod.SPAWN_FOUND}
    assert harness.WORLDS["spawn-fade-wide"] == {"spawn_found": world_mod.SPAWN_FOUND_WIDE}
    noise = harness.scenario("loop", 1, "crowd", 1.0, map_id=harness.TREE, world=("spawn-fade",)).noise
    assert noise.spawn_found == world_mod.SPAWN_FOUND and noise.near_seen == ()


def test_a_monsters_pose_can_read_wrong_in_spells_as_the_templates_do_in_game(model):
    b = plat(model, 360, 300)

    def poses(wrong: float, looks: int = 40) -> list[str]:
        clock = Clock()
        w = World(model, KEYS, (b.id, 240.0), monsters=[("Pig", b.id, 380.0)], origin=clock.now,
                  noise=Noise(pose_wrong=wrong), realism=Realism(respawn_tick=0.0))  # (its own clock: no passes)
        w.invulnerable_ms = 10 ** 9  # it walks into her: no touches
        h = Hunter(FakeEyes(w, clock, model), KEYS, model, log=lambda m: None)
        out = []
        for _ in range(looks):
            out.append(h.look().monsters[0].pose)
            clock.advance_to(clock.now + 0.4)
        return out
    right = poses(0.0)
    assert set(right) == {"move", "stand"}  # (it walks and stops)
    assert all(a != b for a, b in zip(right, poses(1.0)))  # always wrong: walking as standing, and back
    wrong = [a != b for a, b in zip(right, poses(0.25))]
    assert 0.1 <= sum(wrong) / len(wrong) <= 0.45 and poses(0.25) == poses(0.25)  # seeded
    # Drawn once a spell, for each monster.
    w = World(model, KEYS, (b.id, 240.0), monsters=[("Pig", b.id, 380.0), ("Pig", b.id, 300.0)],
              noise=Noise(pose_wrong=0.25))
    eyes = FakeEyes(w, Clock(), model)
    for m in w.monsters:
        draws = []
        for k in range(200):
            w.ms = round(k * 1000 * world_mod.POSE_SPELL_S / 4)
            draws.append(eyes._pose_wrong(m))
        assert all(len(set(draws[i:i + 4])) == 1 for i in range(0, 200, 4))
        assert 0.15 <= sum(draws) / len(draws) <= 0.35


def test_another_player_kills_monsters_and_its_drops_are_not_hers_for_a_while(model):
    c = plat(model, 500, 400)
    w = world(model, c.id, 300.0, monsters=[("Snail", c.id, 310.0)],
              realism=Realism(rival_every=1.0, rival_owns=15.0, rival_takes=20.0))
    w.invulnerable_ms = 100_000
    still(w)
    w.advance(2.0)
    assert [k[3] for k in w.kill_log] == ["rival"] and w.exp == 0
    w.advance(4.0)
    (d,) = w.live_drops()
    assert d.owner == "rival"
    w.x = d.x
    tap(w, 4.0, "z", 0.04)
    w.advance(4.2)
    assert w.picked == {}  # not hers yet
    w.advance(30.0)
    assert d.gone and d.taken_by == "rival"  # the rival picked it up


def test_another_player_hunts_one_place_at_a_time_its_dot_on_the_minimap(model):
    """It kills only what's within RIVAL_REACH of where it stands; with nothing left there it walks to another place
    (RIVAL_SPEED, its minimap dot going along), and kills there once it's there."""
    import math
    from gamelab.localization.players import Players
    c = plat(model, 500, 400)
    clock = Clock()
    w = world(model, c.id, 400.0, monsters=[("Snail", c.id, 100.0), ("Snail", c.id, 700.0)], origin=clock.now,
              realism=Realism(rival_every=1.0, rival_stay=1000.0, respawn=1000.0))
    w.invulnerable_ms = 1_000_000
    still(w)
    eyes, players = FakeEyes(w, clock, model), Players(Localizer(model))
    start = w.rival_at()
    near = min(w.monsters, key=lambda m: abs(m.at(w.t) - start[0]))
    w.advance(clock.now + 1.6)
    first = [k for k in w.kill_log if k[3] == "rival"]
    assert len(first) == 1 and abs(near.died_x - start[0]) <= world_mod.RIVAL_REACH  # the one by it, at once
    away = next(m for m in w.monsters if m is not near)
    walked = []
    for k in range(80):
        clock.advance_to(clock.now + 0.1)
        w.advance(clock.now)
        walked.append(w.rival_at())
        seen = players.update(clock.now, eyes.observe().others)
        if seen:
            assert abs(seen[0].x - walked[-1][0]) < 1  # its dot places it where it is
    second = [k for k in w.kill_log if k[3] == "rival"][1:]
    assert len(second) == 1 and not away.alive and abs(walked[-1][0] - away.died_x) <= world_mod.RIVAL_NEAR
    steps = [abs(b[0] - a[0]) for a, b in zip(walked, walked[1:])]
    assert max(steps) <= world_mod.RIVAL_SPEED * 0.1 + 1e-6  # it walked there, at most 12.5 units a tenth of a second
    gone_at = next(t for t, line in ((float(x.split()[0]), x) for x in w.log) if "gets to platform" in line)
    assert any("another player kills" in line and float(line.split()[0]) >= gone_at for line in w.log)
    assert math.isclose(sum(steps), abs(walked[-1][0] - start[0]), abs_tol=1.0)


def test_monsters_can_start_hurt_their_bars_up_and_come_back_whole(model):
    """Realism.hurt (a scenario's set-up: a monster to finish from the start, loop_tree_grind_attack_key): the monsters
    of a kind there at the start have that share of their HP left, their bars drawn so (the stack reads them); the
    others whole; one killed comes back whole."""
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0, monsters=[("Pig", c.id, 400.0), ("Snail", c.id, 450.0)], facing="right",
                    realism=Realism(hurt=(("pig", 0.25),), respawn_tick=0.0, respawn=2.0)))
    pig, snail = w.monsters
    assert pig.hp == pytest.approx(0.25 * pig.hp_max) and w.drawn_bar(pig) == pytest.approx(0.25)
    assert snail.hp == snail.hp_max and w.drawn_bar(snail) is None
    tap(w, 0.0, "f", 0.04)  # one star (a whole pig takes three: HITS)
    w.advance(1.5)
    assert not pig.alive and w.kills == {"pig": 1}
    w.advance(5.0)  # back 2 s after its death
    assert pig.alive and pig.hp == pig.hp_max and w.drawn_bar(pig) is None
    assert Realism().hurt == ()


def test_potions_run_out_and_arrows_can_be_swallowed_after_an_attack(model):
    c = plat(model, 500, 400)
    w = world(model, c.id, 100.0, hp=0.2, realism=Realism(potions=1))
    tap(w, 0.0, "r", 0.04)
    tap(w, 0.5, "r", 0.04)
    w.advance(1.0)
    assert w.hp == pytest.approx(0.5) and w.potions_used == 1 and any("no potion left" in line for line in w.log)
    for lock, moved in ((False, True), (True, False)):
        w = world(model, c.id, 100.0, realism=Realism(lock_arrows=lock))
        tap(w, 0.0, "f", 0.04)
        tap(w, 0.1, "right", 0.4)  # pressed 0.1 s after the attack's key, let go before 1.0 s: swallowed
        w.advance(2.0)
        assert (w.x > 110) == moved
    assert harness.PRESETS["game"][1].lock_arrows and not harness.PRESETS["plain"][1].lock_arrows


def test_the_knobs_are_off_by_default():
    assert Noise() == Noise(hit_hidden=(), detect_miss=0.0, pose_wrong=0.0)
    assert Realism().respawn == world_mod.RESPAWN and Realism().rival_every == 0 and Realism().potions is None


def test_sim_timeout_isnt_caught_by_a_loop_that_recovers_from_errors():
    clock = Clock(limit=1.0)
    with pytest.raises(SimTimeout):
        try:
            clock.sleep(2.0)
        except Exception:  # what a loop that carries on after errors would do
            pass
    assert not issubclass(SimTimeout, Exception)


def test_thinking_can_be_charged_to_the_fake_clock(monkeypatch):
    real = iter([10.0, 10.5, 10.6])
    monkeypatch.setitem(fakes._REAL, "perf_counter", lambda: next(real))
    clock = Clock(think=2.0)
    clock.mark()        # a fake call ends at real 10.0 ...
    clock.sleep(0.1)    # ... and the next starts at 10.5: 0.5 s of thinking, charged twice over
    assert clock.elapsed == pytest.approx(1.1)
    assert Clock().think == 0.0


def test_with_reads_the_clock_moves_inside_the_stacks_computing(monkeypatch):
    """The harness's --think-clock (Clock.reads, the latency analysis of 2026-11-01): a read of the clock inside the
    stack's computing (monotonic, perf_counter, time) charges what was computed since the last charge first, times
    `think`, so a replan's own computing moves it (navigator.SPLICE_LATE_S can fire). Off (--think alone, or nothing),
    a read there gives the time of the last fake call, whatever was computed since."""
    real = {"t": 10.0}
    monkeypatch.setitem(fakes._REAL, "perf_counter", lambda: real["t"])
    clock = Clock(think=2.0, reads=True)
    clock.mark()       # a fake call ends at real 10.0 ...
    real["t"] = 10.2   # ... and 0.2 s on the stack reads the clock: charged twice over, at once
    t0 = clock.monotonic()
    assert t0 - clock.start == pytest.approx(0.4)
    real["t"] = 10.25
    assert clock.time() - fakes.EPOCH == pytest.approx(0.5) and clock.monotonic() == pytest.approx(t0 + 0.1)
    real["t"] = 10.3
    clock.sleep(0.1)   # (a fake call: what's left charged, then its own time)
    assert clock.elapsed == pytest.approx(0.7)
    for off in (Clock(think=2.0), Clock(reads=True)):
        off.mark()
        real["t"] += 1.0
        assert off.monotonic() == off.start and off.time() == fakes.EPOCH and not off.elapsed
    assert not Clock().reads


def test_the_harness_think_clock_is_think_charged_at_every_read_too():
    """`--think-clock K` (run_one's `think_clock`): the run's clock charges the stack's computing times K at the fakes'
    calls and at every read of it (Scenario.think_reads); not with --think too; not cached; a result says it only
    where it's set, so every run without it is as before to the digit."""
    sc = harness.scenario("loop-grind", 1, "crowd", 5.0, map_id=10002073, think_clock=3.0)
    assert sc.think == 3.0 and sc.think_reads
    plain = harness.scenario("loop-grind", 1, "crowd", 5.0, map_id=10002073, think=1.0)
    assert plain.think == 1.0 and not plain.think_reads
    assert not harness.scenario("loop-grind", 1, "crowd", 5.0, map_id=10002073).think_reads
    with pytest.raises(ValueError):
        harness.scenario("loop-grind", 1, "crowd", 5.0, map_id=10002073, think=1.0, think_clock=1.0)
    from gamelab.sim import baseline
    run = ("loop", 0, "crowd", 5.0, 0.0, "measured", None, 10002073, ("walkers",), (), "choice")
    assert "think_clock" not in baseline.params(harness.run_one, run)  # (off: every cached key as before)
    assert "think_clock" in baseline.uncacheable(baseline.params(harness.run_one, run + (1.0,)))
    p = _parsed(["run", "loop-grind", "--think-clock", "3"])
    assert p.think_clock == 3.0 and p.think == 0.0 and harness._command_line(p)[-2:] == ["--think-clock", "3.0"]
    assert harness._name(p) == "loop-grind-game-think-clock3"  # (not the plain run's name: the review of 2026-11-01)
    assert harness._name(_parsed(["run", "loop-grind", "--think", "1.5"])) == "loop-grind-game-think1.5"
    assert harness._name(_parsed(["run", "loop-grind"])) == "loop-grind-game"
    with pytest.raises(SystemExit):
        _parsed(["run", "loop-grind", "--think", "1", "--think-clock", "3"])


def _parsed(argv: list[str]):
    """The harness's `run` options parsed from `argv` (run's own parser, _run_options)."""
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    harness._run_options(sub.add_parser("run"))
    return p.parse_args(argv)


def _result(seed, epm, share=0.5, hp=0.1, died=None):
    return {"seed": seed, "exp_per_min": epm, "drops_share": share, "hp_lost": hp, "died_at": died}


def test_the_bar_needs_more_exp_on_most_seeds_and_nothing_worse():
    base = [_result(k, 50.0) for k in range(1, 11)]
    ok, lines = harness.compare(base, [_result(k, 60.0) for k in range(1, 11)])
    assert ok and lines[-1] == "PASSES the bar"
    assert not harness.compare(base, [_result(k, 60.0 if k > 3 else 49.0) for k in range(1, 11)])[0]  # 7 of 10
    assert not harness.compare(base, [_result(k, 60.0 if k > 1 else 40.0) for k in range(1, 11)])[0]  # -20%
    assert not harness.compare(base, [_result(k, 60.0, hp=0.2) for k in range(1, 11)])[0]
    assert not harness.compare(base, [_result(k, 60.0, died=100.0 if k == 5 else None) for k in range(1, 11)])[0]


def test_the_bar_refuses_runs_of_two_characters():
    # As runs on two maps: the review's (2026-11-02), a thief's run against a magician's was judged by the bar
    base = [_result(k, 50.0) for k in range(1, 11)]
    hers = [dict(_result(k, 60.0), character="magician") for k in range(1, 11)]
    ok, lines = harness.compare(base, hers)
    assert not ok and lines[0] == "the runs play different characters (['thief'] and ['magician'])"
    assert harness.compare([dict(r, character="magician") for r in base], hers)[0]
    assert harness.compare([dict(r, character="thief") for r in base], [_result(k, 60.0) for k in range(1, 11)])[0]


def test_outcomes_are_read_from_the_world(model):
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0, monsters=[("Snail", c.id, 400.0)], facing="right"))
    tap(w, 0.0, "f", 0.04)
    w.advance(3.0)
    (d,) = w.live_drops()
    w.x = d.x
    tap(w, 3.0, "z", 0.04)
    w.advance(3.5)
    out = harness.outcomes(w, [{"topic": "event", "kind": "backstop", "text": "LOOT_SECONDS: 180 s"}], 60.0)
    assert out["kills"] == 1 and out["exp"] == 2 and out["exp_per_min"] == 2.0
    assert out["drops"] == 1 and out["drops_share"] == 1.0 and out["backstops"] == {"LOOT_SECONDS": 1}
    assert out["longest_idle"] == pytest.approx(60.0 - 3.0, abs=0.6)  # from her pick-up to the end
    # Her potions at the user's reference prices (hunt.HP_PER_MESO, MP_PER_MESO), by what each of hers gives.
    from gamelab.behavior import hunt
    w.potions_used, w.mp_potions = 2, 3
    hp_potion = w._potion() * w.character.max_hp / hunt.HP_PER_MESO
    cash = harness.outcomes(w, [], 60.0)["cash_per_min"]
    assert cash == pytest.approx(out["value_per_min"] - 2 * hp_potion - 3 * w.character.mp_potion / hunt.MP_PER_MESO)
    # ... a Meat (`--world meat`: the user's, 2026-11-01) at what it heals, world.MEAT_HP / 2: 80 mesos.
    assert harness.WORLDS["meat"] == {"potion": world_mod.MEAT_HP, "potions": world_mod.POTIONS}
    assert "potion" in harness.HER_FIELDS and "potions" in {f.name for f in dataclasses.fields(Realism)}
    w.character = dataclasses.replace(w.character, potion=world_mod.MEAT_HP)
    meat = harness.outcomes(w, [], 60.0)["cash_per_min"]
    assert meat == pytest.approx(out["value_per_min"] - 2 * world_mod.MEAT_HP / hunt.HP_PER_MESO
                                 - 3 * w.character.mp_potion / hunt.MP_PER_MESO) and world_mod.MEAT_HP / 2 == 80
    w.drops[0].taken_by, w.drops[0].since = "", 0.0  # a drop left lying its whole life counts as missed
    assert harness.outcomes(w, [], DROP_LIFE + 1)["drops_share"] == 0.0


def test_her_inventorys_tabs_in_the_world_refuse_a_full_tabs_items_and_fill_up(model):
    """Realism.full_tabs and fills_after (the harness's `--world etc-full`, `fills-up`): with her Etc tab full a cap
    under her stays, refused with the game's message (World.pick_refused), while a potion goes into her Use tab, said
    "earned. (Use)", which fills with it; a stand there with nothing she could take is stands_for_nothing."""
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0, drops=[("horny-mushroom-cap", c.id, 100.0)],
                    realism=Realism(full_tabs=("Etc",), fills_after=(("Use", 1),))))
    assert harness.WORLDS["etc-full"] == {"full_tabs": ("Etc",)}
    assert harness.WORLDS["fills-up"] == {"fills_after": (("Etc", 6),)}
    tap(w, 0.5, "z", 0.04)
    w.advance(1.0)
    cap = w.drops[0]
    assert not cap.gone and w.pick_refused >= 1 and w.messages[-1][1] == world_mod.FULL_MESSAGE
    start = 1000.0
    records = [{"topic": "session", "t": start},
               {"topic": "event", "kind": "pick", "t": start + 0.5, "platform": c.id, "x": 100.0}]
    out = harness.outcomes(w, records, 60.0)
    assert out["pick_refused"] == w.pick_refused and out["stands_for_nothing"] == 1
    w.drops.append(dataclasses.replace(cap, kind="red-potion", since=w.t))
    tap(w, 1.5, "z", 0.04)
    w.advance(2.0)
    assert w.drops[-1].taken_by == "her" and w.messages[-1][1].endswith("earned. (Use)") and "Use" in w.full_at
    # The review of 2026-10-31: a full tab still takes a kind she picked into it before (a part-filled stack: v83), not
    # a new kind; the stands for nothing after the hunt had learned the tab full are nothing_known's (before, a stand is
    # how it learns); and `room-later` gives her full tabs room again.
    w.drops.append(dataclasses.replace(cap, kind="red-potion", since=w.t))
    w.drops.append(dataclasses.replace(cap, kind="blue-potion", since=w.t, x=100.0))
    tap(w, 2.5, "z", 0.04)
    tap(w, 2.7, "z", 0.04)
    w.advance(3.0)
    assert [d.taken_by for d in w.drops[-2:]] == ["her", ""] and w.takes("red-potion", w.t)
    assert not w.takes("blue-potion", w.t) and not w.takes("horny-mushroom-cap", w.t) and w.takes("mesos", w.t)
    learned = records + [{"topic": "event", "kind": "inventory", "t": start + 0.2, "tab": "Etc", "full": True}]
    assert harness.outcomes(w, learned, 60.0)["nothing_known"] == 1 and out["nothing_known"] == 0
    assert harness.WORLDS["room-later"] == {"room_at": 180.0}
    w.realism = dataclasses.replace(w.realism, room_at=w.t + 1.0)
    tap(w, w.t + 1.5, "z", 0.04)
    w.advance(w.t + 2.0)
    assert cap.taken_by == "her" and w.room_t is not None and not w.tab_full("Etc", w.t)


def test_her_pets_trail_her_and_take_what_they_walk_over_as_her_pick_ups(model):
    """Realism.pets (the harness's `--world pet-loot`; the user, 2026-11-01: "The pet will pick up whatever thing when
    the pet walks past those loot. But the pet won't proactively go for the loot"): a pet of hers on her platform walks
    to its PET_FOLLOW from her and takes a coin it passes, hers, with the game's line in her pickup log; not one on
    another platform, another player's within its ownership, nor an item of a full tab; on her platform PET_HOP_S after
    she's on another, behind her."""
    from gamelab.sim.world import PET_FOLLOW, PET_HOP_S, PET_SPEED
    assert harness.WORLDS["pet-loot"] == {"pets": 2}
    b, c = plat(model, 360, 300), plat(model, 500, 400)
    w = world(model, c.id, 100.0, drops=[("mesos", c.id, 250.0), ("mesos", b.id, 300.0), ("mesos", c.id, 600.0)],
              realism=Realism(pets=1))
    w.pets = [SimPet(c.id, 300.0, goal=300.0)]  # (200 units off her: it walks to her)
    w.advance(0.5)
    coin, far, other = w.drops
    assert coin.gone and coin.taken_by == "her" and coin.pet and w.messages[-1][1] == "You have gained mesos (+15)"
    assert not far.gone and not other.gone and w.picked == {"mesos": 1}
    assert w.pets[0].x == pytest.approx(300.0 - PET_SPEED * 0.5, abs=PET_SPEED * 0.05 + 1)
    w.advance(3.0)
    assert abs(w.pets[0].x - 100.0) <= PET_FOLLOW[0] + world_mod.PET_WANDER  # (by her: following, then wandering)
    # Another player's coin within its ownership, and her full Etc tab's cap, under the pet: left; a potion taken.
    w = world(model, c.id, 100.0, drops=[("horny-mushroom-cap", c.id, 105.0), ("red-potion", c.id, 95.0)],
              realism=Realism(pets=1, full_tabs=("Etc",)))
    w.drops.append(dataclasses.replace(w.drops[0], kind="mesos", owner="rival", x=100.0))
    w.advance(1.0)
    cap, potion, theirs = w.drops
    assert not cap.gone and potion.taken_by == "her" and potion.pet and not theirs.gone
    assert w.messages[-1][1] == "Red Potion x1 earned. (Use)" and w.pick_refused == 0
    w.advance(w.realism.rival_owns + 0.5)
    assert theirs.taken_by == "her" and theirs.pet and not cap.gone
    # She's on another platform (b): the pet is there PET_HOP_S on, its PET_FOLLOW behind her.
    w = world(model, b.id, 300.0, facing="right", realism=Realism(pets=1))
    w.advance(0.1)
    w.pets[0].platform, w.pets[0].x = c.id, 600.0
    w.advance(0.1 + PET_HOP_S - 0.1)
    assert w.pets[0].platform == c.id
    w.advance(0.1 + PET_HOP_S + 0.1)
    assert w.pets[0].platform == b.id and w.pets[0].x == pytest.approx(300.0 - PET_FOLLOW[0], abs=PET_SPEED * 0.1)


def test_outcomes_tell_her_pets_pick_ups_from_hers_and_where_she_was(model):
    """harness._pets_took: pet_picks, pet_share, pet_value_per_min and her_value_per_min apart, drops_expired (hers
    that lay their whole life), and the tables her pets were fitted by from World.path_log: taken_from (where she was
    as each was taken) and approach (by her closest approach, those taken within APPROACH_TAKEN_S of it)."""
    c = plat(model, 500, 400)
    w = world(model, c.id, 100.0, drops=[("mesos", c.id, 100.0), ("mesos", c.id, 160.0), ("mesos", c.id, 400.0)],
              realism=Realism(pets=1))
    tap(w, 0.0, "z", 0.04)  # (hers under her, at once: before the pet's first step, which takes its own)
    w.advance(0.15)
    w.pets = [SimPet(c.id, 160.0, goal=160.0)]
    w.advance(2.0)
    mine, pets, left = w.drops
    assert mine.taken_by == "her" and not mine.pet and pets.pet and not left.gone
    mine.taken_at = 0.5  # (as if as the path log has her there)
    left.since = -world_mod.DROP_LIFE  # (lain its whole life by now)
    out = harness.outcomes(w, [], 60.0)
    coin = harness.worth_for(model)("mesos")
    assert (out["pet_picks"], out["pet_share"], out["drops_expired"]) == (1, 0.5, 1)
    assert out["pet_value_per_min"] == pytest.approx(coin) and out["her_value_per_min"] == pytest.approx(coin)
    assert out["value_per_min"] == pytest.approx(2 * coin)
    assert out["taken_from"] == {"<=15": 1, "15-45": 0, "45-150": 1, ">150": 0, "off": 0}
    assert out["approach"]["60"] == {"n": 1, "taken": 1} and len(w.path_log) == 20


def test_attacks_chosen_at_a_monster_dead_by_then_are_counted_from_their_records(model):
    """at_dead, casts_at_dead (harness._at_dead): an 'attack' record whose target (its kind, platform and x) had died
    within DEAD_BEFORE_S before it, within DEAD_NEAR of where it was seen; mp_per_kill; dropped, the 'dropped'
    events; mp_throws, the regular attack thrown for her MP (its 'attack' record's `why`)."""
    from gamelab.sim.fakes import EPOCH
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0, monsters=[("Snail", c.id, 300.0)], facing="right"))
    tap(w, 0.0, "f", 0.04)  # one star kills a snail
    w.advance(2.0)
    (died,) = w.death_log
    t, kind, pid, x = died
    assert kind == "snail" and pid == c.id and x == pytest.approx(300.0, abs=30)

    def attack(dt, key="lucky_seven", dx=10.0, on=c.id):
        return {"topic": "attack", "t": EPOCH + t + dt, "key": key, "kind": "snail", "platform": on, "x": x + dx}
    records = [attack(0.4), attack(0.9, "attack"), attack(0.5, dx=60.0), attack(0.5, on=None), attack(3.0),
               attack(-0.2), {"topic": "event", "kind": "dropped"}, {**attack(5.0, "attack"), "why": "MP"}]
    out = harness.outcomes(w, records, 60.0)
    assert (out["at_dead"], out["casts_at_dead"], out["dropped"], out["mp_throws"]) == (2, 1, 1, 1)
    assert out["mp_per_kill"] == 0.0 and out["mp_potions_per_min"] == 0.0


def test_lucky_seven_does_three_stars_a_cast(model):
    """`--world lucky-seven` (world.Her.stars): a pig, 3 of the regular attack's stars (HITS), dies to one cast."""
    c = plat(model, 500, 400)
    for stars, alive in ((1.0, True), (world_mod.LUCKY_SEVEN, False)):
        w = still(world(model, c.id, 100.0, monsters=[("Pig", c.id, 400.0)], facing="right",
                        character=world_mod.Her(stars=stars)))
        tap(w, 0.0, "f", 0.04)
        w.advance(1.5)
        assert w.monsters[0].alive is alive
    assert harness.WORLDS["lucky-seven"] == {"stars": world_mod.LUCKY_SEVEN} and "stars" in harness.HER_FIELDS


def test_her_lucky_seven_key_casts_at_its_mp_and_throws_nothing_without_it(model):
    """world.Her.skill (`--attack`): the Lucky Seven key's attack is Lucky Seven, CAST_MP of her MP (Her.mp) a cast, the
    attack key's still the regular star; with less MP than a cast nothing is thrown and no lock taken; her MP comes back
    at MP_REGEN a second."""
    keys = dataclasses.replace(KEYS, lucky_seven=K("g"))
    her = world_mod.Her(mp=601.0, skill=world_mod.LUCKY_SEVEN)
    c = plat(model, 500, 400)

    def w_of():
        return still(World(model, keys, (c.id, 100.0), monsters=[("Pig", c.id, 400.0)], facing="right",
                           character=her))
    w = w_of()
    tap(w, 0.0, "f", 0.04)  # the regular star: a pig lives on
    w.advance(1.5)
    assert w.monsters[0].alive and w.casts == 0 and w.attacks == 1
    tap(w, 1.5, "g", 0.04)  # a cast: dead
    w.advance(3.0)
    assert not w.monsters[0].alive and w.casts == 1 and w.mp_used == world_mod.CAST_MP
    assert w.mp * 601 == pytest.approx(601 - world_mod.CAST_MP + world_mod.MP_REGEN * 1.5, abs=0.1)
    assert w.casts_nothing == 0
    tap(w, 3.0, "g", 0.04)  # another, with nothing left in its way: MP thrown at nothing (harness: casts_nothing)
    w.advance(4.5)
    assert w.casts == 2 and w.casts_nothing == 1
    w = w_of()
    w.mp = 10 / 601  # under a cast's 11
    tap(w, 0.0, "g", 0.04)
    w.advance(1.5)
    assert w.monsters[0].alive and w.no_mp == 1 and w.casts == 0 and w.attacks == 0 and w.attack_lock_ms <= 0
    assert harness.ATTACK_HER == {"skill": world_mod.LUCKY_SEVEN, "spread": True}


def test_her_mp_bar_shows_a_cast_and_her_pets_drink_the_servers_time_after_its_key(model):
    """World.mp_drawn (what her looks read off her MP bar): a cast's MP and the drink of her pet's it brings show
    MP_SHOWS_MS (0.29 s) after its key, the server's (live a cast's MP was first seen off her bar 0.21-0.53 s after its
    key, a median 0.35); her MP itself is taken at the key, which the game decides a cast on."""
    keys = dataclasses.replace(KEYS, lucky_seven=K("g"))
    her = world_mod.Her(mp=601.0, skill=world_mod.LUCKY_SEVEN, pet_mp=world_mod.PET_MP)
    c = plat(model, 500, 400)
    w = still(World(model, keys, (c.id, 100.0), monsters=[("Pig", c.id, 400.0)], facing="right", character=her))
    w.mp = 40 / 601  # a cast takes her under the pet's warning (30 MP): it drinks
    tap(w, 0.0, "g", 0.04)
    w.advance(0.2)
    assert w.casts == 1 and w.mp_potions == 1 and w.mp > 0.2  # (taken, and the drink, at the key)
    assert w.mp_drawn() == pytest.approx(40 / 601, abs=0.002)  # (her regen meanwhile)
    w.advance(0.1 + world_mod.MP_SHOWS_MS / 1000)
    assert w.mp_drawn() == pytest.approx(w.mp)


def test_a_touch_by_a_monster_just_back_at_its_spawn_point_is_a_fresh_one(model):
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0, monsters=[("Snail", c.id, 300.0)], facing="right",
                    realism=Realism(respawn_tick=0.0)))  # (back on its own clock, RESPAWN after its death)
    tap(w, 0.0, "f", 0.04)  # one star kills a snail
    w.advance(2.0)
    assert not w.monsters[0].alive and not w.touch_log
    w.x = 300.0  # waiting on its spawn point: it comes back on her
    w.advance(2.0 + world_mod.RESPAWN)
    born = w.monsters[0].born
    assert born is not None and w.fresh_log and w.fresh_log == w.touch_log  # every touch then (a second apart)
    assert all(t - born < world_mod.FRESH_S for t, _ in w.fresh_log)


def test_waiting_and_the_touches_then_are_counted_from_the_hunt_loop_s_rules(model):
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0))
    w.touch_log = [(5.0, 0.1), (12.0, 0.05), (13.0, 0.0), (30.0, 0.02)]
    w.fresh_log = [(12.0, 0.05), (13.0, 0.0)]  # (an evaded one doesn't hurt)
    t0 = 1000.0
    records = [{"topic": "session", "t": t0}, {"topic": "hunt", "rule": "shoot", "t": t0 + 1.0},
               {"topic": "hunt", "rule": "wait", "t": t0 + 10.0}, {"topic": "hunt", "rule": "wait", "t": t0 + 11.0},
               {"topic": "hunt", "rule": "shoot", "t": t0 + 15.0}, {"topic": "hunt", "rule": "lull loot", "t": t0 + 25.0}]
    out = harness.outcomes(w, records, 60.0)
    assert out["wait_s"] == pytest.approx(5.0 + 35.0) and out["wait_hp_lost"] == pytest.approx(0.07)
    assert out["fresh_hurt"] == 1 and out["fresh_hp_lost"] == pytest.approx(0.05) and out["alive_s"] == 60.0
    w.died_at = 20.0  # cut at her death
    out = harness.outcomes(w, records, 60.0)
    assert out["alive_s"] == 20.0 and out["wait_s"] == pytest.approx(5.0) and out["wait_hp_lost"] == pytest.approx(0.05)


def test_her_stays_for_drops_and_the_touches_then_are_counted_from_the_hunt_loop_s_rules(model):
    """pile_s, pile_hurt, pile_hp_lost: while a loot rule decides (harness.PILE_RULES) or her objective is a pile (its
    trip there too); stand_hurt and stand_hp_lost: the touches off her trips."""
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0))
    w.touch_log = [(5.0, 0.1), (12.0, 0.05), (13.0, 0.0), (21.0, 0.02), (30.0, 0.04)]
    t0 = 1000.0
    records = [{"topic": "session", "t": t0}, {"topic": "hunt", "rule": "shoot", "t": t0 + 1.0},
               {"topic": "hunt", "rule": "urgent pile", "objective": "pile on 23 at x 104", "t": t0 + 10.0},
               {"topic": "chain", "t": t0 + 10.5}, {"topic": "moved", "t": t0 + 12.5},
               {"topic": "hunt", "rule": "travel", "objective": "pile on 23 at x 104", "t": t0 + 11.0},
               {"topic": "hunt", "rule": "shoot", "objective": None, "t": t0 + 15.0},
               {"topic": "hunt", "rule": "pile here", "t": t0 + 20.0}, {"topic": "hunt", "rule": "wait", "t": t0 + 22.0}]
    out = harness.outcomes(w, records, 60.0)
    assert out["pile_s"] == pytest.approx(5.0 + 2.0) and out["pile_hurt"] == 2
    assert out["pile_hp_lost"] == pytest.approx(0.07)
    assert out["trip_hurt"] == 1 and out["stand_hurt"] == 3 and out["stand_hp_lost"] == pytest.approx(0.16)


def test_a_knock_onto_a_lower_platform_is_counted_as_a_knock_off(model, monkeypatch):
    """World.knock_log (sim.harness: knock_falls): a touch whose knock lands her on another platform more than
    KNOCK_FALL_DY lower; one that leaves her on hers isn't one. (Her touches all take HP here: a MISS knocks nothing.)"""
    monkeypatch.setattr(world_mod, "evade_chance", lambda info, her=None, mob_level=None: 0.0)
    b = plat(model, 360, 300)
    w = world(model, b.id, 375.0, monsters=[("Snail", b.id, 360.0)])
    w.monsters[0].speed = 0.0
    w.advance(2.0)
    assert w.touches >= 1 and w.plat is not None and w.plat.id != b.id
    assert len(w.knock_log) == 1 and w.knock_log[0][1:3] == (b.id, w.plat.id) and w.knock_log[0][3] > 100
    out = harness.outcomes(w, [], 60.0)
    assert out["knock_falls"] == 1 and out["knock_fall_dy"] == w.knock_log[0][3]
    c = plat(model, 500, 400)
    w = world(model, c.id, 375.0, monsters=[("Snail", c.id, 360.0)])
    w.monsters[0].speed = 0.0
    w.advance(2.0)
    assert w.touches >= 1 and w.knock_log == []  # knocked along the floor


def test_a_run_can_hunt_only_some_grounds():
    """`--grounds 23`: the policy's tours hunt only those grounds (all of them if the value map has none of them)."""
    from types import SimpleNamespace
    got = []

    class Sim:
        def tour(self, *a, **kw):
            return SimpleNamespace(hunting_grounds=lambda visit_s=None: [1, 2, 23])

    def policy(sim):
        got.append(sim.tour().hunting_grounds())
    harness._only(policy, (23,))(Sim())
    harness._only(policy, (99,))(Sim())
    assert got == [[23], [99]]
    sc = harness.scenario("loop", 1, "crowd", 1.0, map_id=harness.TREE, grounds=(23,))
    assert sc.name == "loop:crowd:1:10002073:g23" and sc.fn is not harness.POLICIES["loop"]


def test_the_safety_stop_is_counted_and_the_bar_compares_the_seeds_both_have(model):
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0))
    w.touch_log = [(5.0, 0.014), (12.0, 0.02), (13.0, 0.0), (30.0, 0.01)]  # (t, the HP it took: 0 evaded)
    t0 = 1000.0
    records = [{"topic": "session", "t": t0}]
    out = harness.outcomes(w, records, 60.0)
    assert (out["safety_stops"], out["safety_at"]) == (0, None)
    assert out["hp_lost"] == pytest.approx(0.044)
    assert out["alive_s"] == 60.0
    records.append({"topic": "event", "kind": "safety_stop", "t": t0 + 20.0, "text": "HP 31%, under 65% for 2.0 s"})
    w.ms = 25_000  # (the world's time once the run's over: her retreat after the stop)
    out = harness.outcomes(w, records, 60.0)
    assert (out["safety_stops"], out["safety_at"]) == (1, 20.0)
    # The run's time ends with the hunt (her retreat after the stop): not idle to the horizon. The touches to then.
    assert out["alive_s"] == 25.0 and out["longest_idle"] <= 25.0 and out["stuck_s"] == 0.0
    assert harness._mean([out, dict(out, safety_at=None)], "safety_at") == 1  # runs it stopped, as deaths are counted
    assert harness._mean([dict(out, darts_at=40.0), out], "darts_at") == 1
    assert not any(k.startswith("rest") for k in out)  # the rests went on 2026-10-26
    base = [_result(k, 50.0) for k in range(1, 11)]
    # With no seed in common, never a pass.
    assert not harness.compare([], [])[0] and not harness.compare(base, [_result(k, 60.0) for k in range(11, 21)])[0]
    # The bar is over the seeds both have: a 20-seed base against 10 seeds of a candidate compares those 10.
    harder = base + [dict(_result(k, 50.0, share=0.1, hp=1.0)) for k in range(11, 21)]
    assert not harness.compare(harder, [_result(k, 60.0, share=0.4, hp=0.2) for k in range(1, 11)])[0]
    # Fewer than 10 seeds in common never pass; a death counts on any seed, compared or not.
    better = [dict(r, exp_per_min=60.0) for r in base]
    assert not harness.compare(base[:9], better[:9])[0]
    assert harness.compare(base, better)[0]
    died = better + [dict(better[0], seed=15, died_at=600.0)]
    assert not harness.compare(base, died)[0]
    # A stop (the safety stop, out of darts) counts against it as a death does: the hunt ended.
    for key in ("safety_stops", "darts_stops"):
        ok, lines = harness.compare(base, [dict(r, **{key: 1}) if r["seed"] == 3 else r for r in better])
        assert not ok and "the hunt stopped (the safety stop, out of darts): seed(s) 3" in lines


def test_the_touches_on_her_trips_are_counted_from_a_chains_or_moves_start_to_its_end(model):
    # harness._travelling: from each 'chain' or 'move' record (a chain sent, a move made alone; a chain queued on the
    # move is in the first one's) to the next 'moved' record, in world time; the touches that hurt her then.
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0))
    w.touch_log = [(2.0, 0.1), (5.5, 0.05), (6.0, 0.0), (9.0, 0.2), (14.0, 0.01)]
    t0 = 1000.0
    records = [{"topic": "session", "t": t0}, {"topic": "chain", "t": t0 + 5.0}, {"topic": "chain", "t": t0 + 6.5},
               {"topic": "moved", "t": t0 + 7.0}, {"topic": "move", "t": t0 + 13.0}, {"topic": "moved", "t": t0 + 15.0}]
    out = harness.outcomes(w, records, 60.0)
    assert out["trip_s"] == pytest.approx(4.0) and out["trip_hurt"] == 2 and out["trip_hp_lost"] == pytest.approx(0.06)
    # A move the mover can't do from where she stands ends at its 'cannot' event: standing there isn't travel.
    records[4:5] = [{"topic": "move", "t": t0 + 8.0}, {"topic": "event", "kind": "cannot", "t": t0 + 8.1},
                    {"topic": "move", "t": t0 + 13.0}]
    out = harness.outcomes(w, records, 60.0)
    assert out["trip_s"] == pytest.approx(4.1) and out["trip_hurt"] == 2


def test_her_trips_touches_again_and_by_what_she_was_doing_are_counted(model):
    # harness._retouches: a touch (a MISS too) 2 s or less after the one before; _trips: her trips ('trip' records) and
    # those that got there, the time on the move an arrival, her touches by what she was doing (a chain's plan, a move
    # of the mover's, neither), the mover's moves and the navigator's evade steps by kind.
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0))
    w.touch_log = [(2.0, 0.1), (3.4, 0.0), (5.5, 0.05), (6.0, 0.0), (9.0, 0.2), (14.0, 0.01), (15.9, 0.0)]
    t0 = 1000.0
    records = [{"topic": "session", "t": t0}, {"topic": "chain", "t": t0 + 5.0}, {"topic": "chain", "t": t0 + 6.5},
               {"topic": "moved", "t": t0 + 7.0}, {"topic": "trip", "t": t0 + 7.1, "result": "arrived"},
               {"topic": "event", "kind": "evade", "step": "spot", "t": t0 + 8.0},
               {"topic": "move", "t": t0 + 13.0}, {"topic": "moved", "t": t0 + 15.0},
               {"topic": "trip", "t": t0 + 15.1, "result": "timeout"}, {"topic": "trip", "t": t0 + 70.0, "result": "arrived"}]
    out = harness.outcomes(w, records, 60.0)
    assert out["retouch_2s"] == 3  # (3.4 after 2.0, 6.0 after 5.5, 15.9 after 14.0)
    assert out["trips"] == {"total": 2, "arrived": 1} and out["arrive_share"] == 0.5  # (the one past the horizon: not)
    assert out["trip_s_per_arrival"] == pytest.approx(4.0)
    assert out["touches_by"] == {"chain": 2, "move": 1, "off": 4} and out["mover_moves"] == 1
    assert out["evades"] == {"spot": 1}
    assert "walk" in harness.POLICIES and list(harness.WALK_PLATFORMS) == list(range(7, 34))


def test_her_box_can_be_the_clients_in_the_simulated_world(model):
    # Realism.her_box (`--world client-body`): 22 either side of her feet and 65 tall (motion.bodies), not the
    # simulator's touch test (world.HER_TOUCH, the client's: a line 50 tall): a slime walking on B, 140 above C's
    # floor, never reaches her there; one on a platform 60 over her does with the planner's box only (her head 65 up).
    from gamelab.motion import bodies
    assert world_mod.HER_BOX_CLIENT == (bodies.HER_HALF, bodies.HER_TALL) == (22.0, 65.0)
    assert harness.WORLDS["client-body"] == {"her_box": world_mod.HER_BOX_CLIENT}
    sc = harness.scenario("loop", 1, "game", 1.0, world=("client-body",))
    assert sc.realism.her_box == (22.0, 65.0) and Realism().her_box is None
    assert harness.WORLDS["wide-body"] == {"her_box": (15.0, 55.0)}  # (as the world had it until 2026-11-02)
    t = tree()
    t["foothold"]["1"]["9"] = {"30": {"x1": 300, "y1": 440, "x2": 500, "y2": 440, "prev": 0, "next": 0}}
    m = mapdata.parse(1, t, "Test Map", {}, {"7": "Slime"}, {"7": 3})
    c, over = plat(m, 500, 400), plat(m, 440, 400)
    for box, touched in ((None, 0), ((22.0, 65.0), 1)):
        w = still(world(m, c.id, 400.0, monsters=[("Slime", over.id, 400.0)], realism=Realism(her_box=box)))
        for mo in w.monsters:
            mo.speed = 0.0
        w.advance(0.5)
        assert w.touches == touched, box


def test_a_short_harness_run_on_the_real_map():
    if not scenarios.available(harness.TREE):
        pytest.skip("needs Tree Dungeon's data")
    from gamelab.motion import route
    out = harness.run_one("tour", 1, "game", minutes=1.0, map_id=harness.TREE)
    assert out["policy"] == "tour" and "SimTimeout" in out["result"] and out["exp_per_min"] >= 0
    assert set(harness.COLUMNS) - {"seed"} <= set(out) and out["ways"] == route.ROUTES
    # (--ways: route.ROUTES for the run, put back)
    ways = harness.run_one("tour", 1, "game", minutes=1.0, ways=3, map_id=harness.TREE)
    assert ways["ways"] == 3 and route.ROUTES == out["ways"]


def test_a_harness_run_takes_her_looks_from_the_feed_with_todays_look_costs():
    if not scenarios.available(harness.TREE):
        pytest.skip("needs Tree Dungeon's data")
    from gamelab.sim import fakes
    coded = fakes.LOOK_S
    # (a minute: with her place the world's, fakes.TRUE_PLACE, she hunts where she starts past half a minute, no trip;
    # seed 3 since work/replan's loot work on master 072e1ec's monsters, 2026-10-31: on seeds 1 and 2 no glance in the
    # minute; on 3-8 50-211)
    out = harness.run_one("loop", 3, "game", minutes=1.0, costs="measured", map_id=harness.TREE)
    assert out["feed"] and out["costs"] == "measured" and fakes.LOOK_S == coded  # put back after the run
    rec = out["feed_record"]
    assert rec["frames"] > 0 and rec["senses"]["looks"] > 0 and out["looks"] == rec["senses"]["looks"]
    assert out["nav"]["glances"] == rec["senses"]["glances"] > 0


def test_the_stacks_switches_a_world_sets_are_the_runs_alone(monkeypatch):
    # `--world hang-2s` (control.navigator.HANG_LONGER and HANG_TAKE off: the 2 s hang alone), `hang-rules` (both on,
    # the default: the user's "keep the rope rule on", 2026-11-02) and `no-hang` (ROPE_HANG off): the stack's switches,
    # set by run_one around the run and put back after, the world as the preset has it
    if not scenarios.available(harness.TREE):
        pytest.skip("needs Tree Dungeon's data")
    from gamelab.control import navigator
    seen = []
    monkeypatch.setattr(scenarios, "run_sim", lambda sc: seen.append(
        (navigator.HANG_LONGER, navigator.HANG_TAKE, navigator.ROPE_HANG, sc.realism)) or 1 / 0)
    on = (navigator.HANG_LONGER, navigator.HANG_TAKE, navigator.ROPE_HANG)
    assert on == (True, True, True)
    for world in (("hang-2s",), ("hang-rules",), ("no-hang", "zero-walls")):
        with pytest.raises(ZeroDivisionError):
            harness.run_one("loop", 1, "game", minutes=0.5, map_id=harness.TREE, world=world)
        assert (navigator.HANG_LONGER, navigator.HANG_TAKE, navigator.ROPE_HANG) == on  # (put back)
    assert [s[:3] for s in seen] == [(False, False, True), (True, True, True), (True, True, False)]
    game = harness.PRESETS["game"][1]
    assert seen[0][3] == seen[1][3] == game and seen[2][3] == dataclasses.replace(game, zero_walls=True)


def test_the_feeds_costs_as_measured_at_1366x768_are_the_runs_alone(monkeypatch):
    # --costs measured-1366: the fakes' look costs as `measured`'s, and the feed made for the run takes
    # feed.COSTS_1366 (SyncFeed's costs are COSTS as it is when it's made); both put back after the run.
    if not scenarios.available(harness.TREE):
        pytest.skip("needs Tree Dungeon's data")
    from gamelab.localization import feed
    from gamelab.sim import fakes
    seen = []
    monkeypatch.setattr(scenarios, "run_sim", lambda sc: seen.append((feed.COSTS, fakes.LOOK_S)) or 1 / 0)
    coded, look = feed.COSTS, fakes.LOOK_S
    with pytest.raises(ZeroDivisionError):
        harness.run_one("loop", 1, "game", minutes=0.5, costs="measured-1366", map_id=harness.TREE)
    assert seen == [(feed.COSTS_1366, harness.COSTS["measured"]["LOOK_S"])]
    assert feed.COSTS is coded and fakes.LOOK_S == look
    # (since 2026-10-30 COSTS are the one pass's at 1366x768, the live size: COSTS_1366 is them, the whole screen 8.4
    # ms in every frame's pass where 1920x1080's took 100 on its thread)
    assert feed.COSTS_1366 is feed.COSTS and feed.COSTS.wide < 0.01
    made = feed.SyncFeed.__new__(feed.SyncFeed)
    monkeypatch.setattr(feed, "COSTS", feed.COSTS_1366)
    made.__init__(type("Step", (), {"senses": None})(), lambda ropes: None)
    assert made.costs is feed.COSTS_1366


def test_stretches_are_timed_from_a_mark_to_the_next_charge():
    from gamelab.sim import latency
    s = latency.Stretches()
    s.charge("send")  # nothing marked yet: nothing timed
    s.mark()
    s.charge("send")
    s.charge("send")  # charged once per mark
    s.mark()
    s.charge("_now")
    assert [len(s.times[k]) for k in ("send", "_now")] == [1, 1]
    assert [line.split()[0] for line in s.table()[1:]] == ["a", "a", "any"]


def test_her_trips_are_counted_from_the_stacks_log(model):
    """navigation: plans cut short by a hit and off the path, replans (a cut and the landing it logs next are one),
    misses (landed elsewhere than planned) and the falls among them (onto a lower platform, or onto none)."""
    from gamelab.control.mover import platform_text
    a, b, c = plat(model, 300, 100), plat(model, 360, 300), plat(model, 500, 400)  # c the lowest (y is down)
    log = [f"  4.0     hit (90% -> 80%): plan cut short; on {platform_text(b)} at x 300 after 0.40 s: replanning from there",
           f"  4.5     hit on the way (90% -> 80%): on {platform_text(b)}; replanning",  # (the same replan)
           f"  5.0   {platform_text(b)} at x 300 (dot): jump->4  [2 moves to go; chain ended: end]",
           f"  6.0     off the path (on {c.id}, planned {b.id}): plan cut short; on {platform_text(c)} at x 400 after "
           "0.30 s: replanning from there",
           f"  6.4     ended on {platform_text(c)} instead of platform {b.id}; replanning",  # the same replan: a fall
           f"  7.0     ended on {platform_text(a)} instead of platform {b.id}; replanning",  # up instead: a miss
           f"  8.0     ended on air/rope instead of platform {c.id}; replanning",                # a fall
           f"  9.0     hit on the way (80% -> 70%): on {platform_text(c)}; replanning"]
    assert harness.navigation(log, model) == {"cut_hit": 1, "cut_off_path": 1, "replans": 5, "misses": 3, "falls": 2,
                                              "cut_touch": 0, "cut_miss": 0,  # (the log's cut is a hit's of before)
                                              "immune_spent": 0}


def test_a_world_drops_what_its_maps_monsters_drop(model):
    """The harness's `game-drops` (Realism.drops MAP_DROPS): a monster seen in game drops as it did there (GAME_DROPS),
    and its other drops (perception.drops) at OSMLib's reference rates; one not seen in game its drops at those rates
    (an item with none as its rarest), and mesos."""
    from types import SimpleNamespace
    from gamelab.perception.drops import MONSTERS
    from gamelab.sim.world import GAME_DROPS, MAP_DROPS, MESOS_SHARE, NO_RATE, game_drops

    def a_map(map_id, *names):
        return SimpleNamespace(id=map_id, spawns=[SimpleNamespace(name=n) for n in names])

    tree = game_drops(a_map(10002073, "Horny Mushroom", "Slime", "Horny Mushroom"))
    assert tree[:3] == (("horny-mushroom", "mesos", MESOS_SHARE), ("horny-mushroom", "horny-mushroom-cap", 0.40),
                        ("horny-mushroom", "orange-potion", 0.01))
    # The Magic Potion, which OSMLib gives no rate for: as rarely as a Horny Mushroom's rarest, its Topaz Ore (it had
    # been 1%, five times that)
    assert ("horny-mushroom", "magic-potion", 0.002) in tree and ("horny-mushroom", "sapphire-ore", 0.002) in tree
    # The slime as seen in game (Squishy Liquid 65%, where OSMLib gives 40%), and its ores, not seen there, at OSMLib's
    assert [r for r in tree if r[0] == "slime"] == [r for r in GAME_DROPS if r[0] == "slime"] + [
        ("slime", "orihalcon-ore", 0.002), ("slime", "sapphire-ore", 0.003)]
    assert len(tree) == 1 + 7 + 4 + 2
    # A monster neither knows: mesos alone; one with no rate given at all: NO_RATE.
    assert game_drops(a_map(5, "Jr. Necki")) == (("jr.-necki", "mesos", MESOS_SHARE),)
    assert NO_RATE <= min(r for m in MONSTERS.values() for r in m.rates.values())
    # A world on this made-up map, of snails (the first map's, seen in game): what they dropped there.
    w = world(model, plat(model, 360, 300).id, 240.0, realism=Realism(drops=MAP_DROPS))
    assert w._drop_table == tuple(r for r in GAME_DROPS if r[0] == "snail")
    assert harness.PRESETS["game-drops"][1].drops == MAP_DROPS


def test_her_stars_and_casts_can_be_her_own_as_measured_on_a_map(model):
    """world.Her.star, skill_star, cast_mp, mp_potion (LEVEL_25: her stars as read off the bars at level 25, Lucky
    Seven at level 16): with `spread`, a regular star does 20-140 before defence and a cast two of 96-192, its MP 14; her
    pet's MP potion gives 100."""
    keys = dataclasses.replace(KEYS, lucky_seven=K("g"))
    her = world_mod.Her(mp=623.0, skill=world_mod.LUCKY_SEVEN, spread=True, star=(20, 140), skill_star=(96, 192),
                        cast_mp=14.0, mp_potion=100.0, pet_mp=0.05)
    c = plat(model, 500, 400)
    w = still(World(model, keys, (c.id, 100.0), monsters=[("Pig", c.id, 400.0)], facing="right", character=her))
    assert 20 <= w._damage(1.0, False) * world_mod.STAR_MEAN <= 140
    assert 192 <= w._damage(world_mod.LUCKY_SEVEN, False) * world_mod.STAR_MEAN <= 384
    w.mp = 20 / 623
    assert w._cast() and w.mp_used == 14.0 and w.mp * 623 == pytest.approx(6 + 100, abs=0.01)  # (the pet drank)


def test_the_maps_pass_fills_it_to_its_cap_at_random_points(model):
    """Realism.respawn_tick (`--world tick-respawn`, the game's respawn as measured on Tree Dungeon): a dead monster
    isn't back on its own clock: at the map's passes the map is filled to its cap (respawn_cap of its points, whoever
    else is there), at points drawn at random among the clear ones, once a pass at most; the map starts at its cap.
    Off (0), each is back `respawn` s after it died, as before. (Six points 125 units apart: none keeps another.)"""
    c = plat(model, 500, 400)
    snails = [("Snail", c.id, 60.0 + 125 * k) for k in range(6)]
    w = world(model, c.id, 50.0, monsters=snails, realism=Realism(respawn_tick=7.0, respawn_cap=0.5))
    w.invulnerable_ms = 10 ** 9
    still(w)
    assert sum(m.alive for m in w.monsters) == 3  # at its cap from the start
    born, killed = [], 0
    for k in range(60):  # every 3.5 s: whatever is alive killed by another player
        alive = [m for m in w.monsters if m.alive]
        for m in alive:
            w._others_kill(m, w._camp_rng, 1.0)
        killed += len(alive)
        w.advance(3.5 * (k + 1))
        born += [m.born for m in w.monsters if m.alive]
        assert sum(m.alive for m in w.monsters) <= 3
    phases = {round(t % 7.0, 1) for t in born}
    assert len(phases) == 1  # only at the passes
    alive = sum(m.alive for m in w.monsters)
    assert w.respawn_ticks == 30 and w.respawn_births == len(born) == killed + alive - 3  # it made what had died ...
    assert all(n == 3 for n in Counter(born).values())  # ... the map back at its cap every pass
    # Nothing dies: nothing comes back, however many points are free.
    w = world(model, c.id, 50.0, monsters=snails, realism=Realism(respawn_tick=7.0, respawn_cap=0.5))
    still(w)
    w.advance(70.0)
    assert w.respawn_ticks == 10 and w.respawn_births == 0 and sum(m.alive for m in w.monsters) == 3
    # Another player there changes no cap.
    w = world(model, c.id, 50.0, monsters=snails, realism=Realism(respawn_tick=7.0, respawn_cap=0.5,
                                                                  rival_every=1e6))
    assert sum(m.alive for m in w.monsters) == 3


def test_the_maps_cap_is_the_servers_formula(model):
    """MapModel.mob_cap (the server's LifePool, as OrionAlpha has it): one monster in 128,000 square px of the box about
    the map's footholds (each 30 inside its ends, 300 over its top, 10 under it, 10 more each side; at least the
    800 x 600 screen, its height less 450) times its mobRate, 1 to 40; more only with more players than half of it, up
    to twice. The simulator's pass fills the map to it by default (Realism.respawn_cap None), her and the other
    players counted; a share of the points is a knob."""
    # The test map: 0..800 x 300..500, a box of 760 x 530: the screen's 800 x 600, 3.75.
    assert model.mob_cap() == 3 and model.mob_cap(2) == 3 and model.mob_cap(5) == 5 and model.mob_cap(6) == 6
    c = plat(model, 500, 400)
    snails = [("Snail", c.id, 100.0 + 60 * k) for k in range(10)]
    w = world(model, c.id, 50.0, monsters=snails, realism=Realism(respawn_tick=7.0))
    assert sum(m.alive for m in w.monsters) == 3
    # The maps': Tree Dungeon II 22 (its 31 points), Tree Dungeon III 20 (its mobRate 0.9); Tree Dungeon II with 20
    # players 28, 44 or more 44.
    for map_id, cap in ((10002073, 22), (10002074, 20)):
        if scenarios.available(map_id):
            assert mapdata.load(map_id).mob_cap() == cap
    if scenarios.available(10002073):
        tree = mapdata.load(10002073)
        assert [tree.mob_cap(u) for u in (11, 12, 20, 44, 60)] == [22, 22, 28, 44, 44]


def test_the_maps_pass_fills_only_points_with_no_monster_within_100_units(model):
    """The server's rule (OrionAlpha's LifePool.tryCreateMob; the user, 2026-10-31): a pass fills a spawn point only
    with no live monster within SPAWN_CLEAR (100 units) of it across and up or down, nor a point found clear before it
    in the pass (the points in the map's order), the cap aside. Six snail points 60 units apart: an empty map's pass
    fills the first, third and fifth, and the map stays at three under a cap of six; a kill frees its own point (the
    nearest monster 120 off) and the next pass fills it, its neighbour 60 off kept by it."""
    c = plat(model, 500, 400)
    snails = [("Snail", c.id, 100.0 + 60 * k) for k in range(6)]
    w = still(world(model, c.id, 50.0, monsters=snails, realism=Realism(respawn_tick=7.0, respawn_cap=1.0)))
    w.invulnerable_ms = 10 ** 9
    while w.respawn_ticks == 0:  # (the map's phase its own: kill at a pass, the deaths over by the next)
        w.advance(w.t + 0.1)
    for m in w.monsters:
        w._others_kill(m, w._camp_rng, 1.0)
    w.advance(w.t + 7.05)
    assert sorted(m.spawn for m in w.monsters if m.alive) == [0, 2, 4] and w.respawn_births == 3
    w.advance(w.t + 70.0)  # ten passes: every other point within 60 of a monster, nothing more
    assert sorted(m.spawn for m in w.monsters if m.alive) == [0, 2, 4] and w.respawn_births == 3
    assert w.respawn_blocked >= 30
    first = next(m for m in w.monsters if m.alive and m.spawn == 0)
    w._others_kill(first, w._camp_rng, 1.0)
    w.advance(w.t + 7.05)
    assert sorted(m.spawn for m in w.monsters if m.alive) == [0, 2, 4] and w.respawn_births == 4
    assert first.alive and len(w.monsters) == 6  # (its own monster back, nothing new made)
    # Off (respawn_tick 0, the default until 2026-10-31): back 8 s after its death, every one.
    w = world(model, c.id, 50.0, monsters=snails[:1], realism=Realism(respawn_tick=0.0))
    w._others_kill(w.monsters[0], w._camp_rng, 1.0)
    w.advance(7.9)
    assert not w.monsters[0].alive
    w.advance(8.1)
    assert w.monsters[0].alive


def test_rope_tops_counts_her_climbs_and_the_touches_on_a_rope_or_at_its_top_just_after_one():
    """sim.harness `rope_tops` (World.climb_log, grab_log, touch_place): a touch on a rope is one, standing within
    ROPE_TOP_DX of a rope she climbed over ROPE_TOP_S or less before is one, one further off or later isn't; a MISS
    counts, not as hurt; a grab from a rope's top (a climb down) isn't a climb set off on."""
    from types import SimpleNamespace
    w = SimpleNamespace(climb_log=[(10.0, -80.0, 10), (50.0, 111.0, 18)],
                        grab_log=[(8.6, -80.0, 158.0), (48.0, 111.0, 175.0), (55.0, 111.0, 1.0), (65.0, -8.0, 76.0)],
                        touch_place=[(9.5, "rope", -80.0, -1430.0),      # on the rope (before it climbed over)
                                     (11.0, "ground", -70.0, -1469.0),   # at the top 1 s on
                                     (11.5, "ground", -150.0, -1480.0),  # on 10, but off the rope's top
                                     (13.0, "ground", -80.0, -1469.0),   # 3 s on
                                     (51.0, "air", 120.0, -900.0),       # knocked off the top of rope 4 (a MISS)
                                     (70.0, "rope", 111.0, -850.0)],     # past `end`
                        touch_log=[(9.5, 0.12), (11.0, 0.13), (11.5, 0.1), (13.0, 0.1), (51.0, 0.0), (70.0, 0.1)])
    assert harness._rope_tops(w, 60.0) == {"climbs": 2, "grabs": 2, "touched": 3, "on_rope": 1, "hurt": 2}
    assert harness._rope_tops(w, 30.0) == {"climbs": 1, "grabs": 1, "touched": 2, "on_rope": 1, "hurt": 2}


def test_rope_tops_counts_her_hangs_under_a_rope_s_top_from_their_events():
    """sim.harness rope_tops' hangs (control.navigator.ROPE_HANG's 'rope_hang' events): one at each hang's start, the
    seconds from their ends, how they ended (up regardless after HANG_S, the way up clear, up at a longer wait's end,
    the touch taken, touched), and those that went longer than HANG_S (their 'longer' event, no seconds of its own)."""
    events = [{"kind": "rope_hang", "step": "hang"}, {"kind": "rope_hang", "step": "clear", "s": 1.24},
              {"kind": "rope_hang", "step": "hang"}, {"kind": "rope_hang", "step": "regardless", "s": 4.02},
              {"kind": "rope_hang", "step": "hang"}, {"kind": "rope_hang", "step": "longer", "at": 0.5},
              {"kind": "rope_hang", "step": "waited", "s": 2.6},
              {"kind": "rope_hang", "step": "hang"}, {"kind": "rope_hang", "step": "took", "s": 0.0},
              {"kind": "climb_on"}]
    assert harness._hangs(events) == {"hangs": 4, "hang_s": 7.9, "hang_regardless": 1, "hang_clear": 1,
                                      "hang_waited": 1, "hang_took": 1, "hang_touched": 0, "hang_longer": 1}
    assert harness._hangs([]) == {"hangs": 0, "hang_s": 0.0, "hang_regardless": 0, "hang_clear": 0,
                                  "hang_waited": 0, "hang_took": 0, "hang_touched": 0, "hang_longer": 0}
