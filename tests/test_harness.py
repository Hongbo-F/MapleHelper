"""The simulator's realism knobs (all off by default) and the results harness that compares hunting
policies on the same seeds (gamelab.sim.harness). Synthetic maps need no game data; the real-map test is
skipped without it."""

import dataclasses

import pytest

from gamelab.behavior.hunter import Hunter
from gamelab.localization import mapdata
from gamelab.localization.pose import Localizer
from gamelab.sim import fakes, harness, scenarios
from gamelab.sim import world as world_mod
from gamelab.sim.fakes import Clock, FakeEyes, SimTimeout
from gamelab.sim.world import DROP_LIFE, Noise, Realism, World

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


def test_near_her_on_her_platform_a_look_finds_a_monster_as_seldom_as_live(model):
    """`--world near-misses` (Noise.near_seen, world.NEAR_SEEN, OFF_SEEN; the live hunt of 2026-10-25): a monster on
    her platform within 25 units of her is never found, within 50 found by 7% of looks, within 100 by 34%, within 150
    by 55%, farther by 60%; one on another platform by 70% (Noise.off_seen; without it, as the preset's detect_miss
    has it)."""
    c = plat(model, 500, 400)
    clock = Clock()

    def found(x: float, pid: int = c.id, looks: int = 400, off: float | None = None) -> float:
        w = still(World(model, KEYS, (c.id, 100.0), monsters=[("Snail", pid, x)], origin=clock.now,
                        noise=Noise(near_seen=world_mod.NEAR_SEEN, off_seen=off)))
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
    assert found(300.0, b.id, off=world_mod.OFF_SEEN) == pytest.approx(0.7, abs=0.07)  # ... or off_seen
    assert harness.WORLDS["near-misses"] == {"near_seen": world_mod.NEAR_SEEN, "off_seen": world_mod.OFF_SEEN}
    assert {"near_seen", "off_seen"} <= set(harness.NOISE_FIELDS)
    noise = harness.scenario("loop", 1, "crowd", 1.0, map_id=harness.TREE, world=("near-misses",)).noise
    assert (noise.near_seen, noise.off_seen) == (world_mod.NEAR_SEEN, world_mod.OFF_SEEN)


def test_a_monsters_pose_can_read_wrong_in_spells_as_the_templates_do_in_game(model):
    b = plat(model, 360, 300)

    def poses(wrong: float, looks: int = 40) -> list[str]:
        clock = Clock()
        w = World(model, KEYS, (b.id, 240.0), monsters=[("Pig", b.id, 380.0)], origin=clock.now,
                  noise=Noise(pose_wrong=wrong))
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


def test_the_knobs_are_off_by_default_and_spawn_share_thins_the_monsters():
    assert Noise() == Noise(hit_hidden=(), detect_miss=0.0, pose_wrong=0.0)
    assert Realism().respawn == world_mod.RESPAWN and Realism().rival_every == 0 and Realism().potions is None
    if not scenarios.available():
        pytest.skip("needs the real map's data")
    model = scenarios.real_map()
    full = World(model, KEYS, (36, 300.0), monsters="all")
    half = World(model, KEYS, (36, 300.0), monsters="all", realism=Realism(spawn_share=0.5))
    assert len(full.monsters) == 48 and 14 <= len(half.monsters) <= 34


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
    w.drops[0].taken_by, w.drops[0].since = "", 0.0  # a drop left lying its whole life counts as missed
    assert harness.outcomes(w, [], DROP_LIFE + 1)["drops_share"] == 0.0


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
    w = still(world(model, c.id, 100.0, monsters=[("Snail", c.id, 300.0)], facing="right"))
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


def test_a_knock_onto_a_lower_platform_is_counted_as_a_knock_off(model):
    """World.knock_log (sim.harness: knock_falls): a touch whose knock lands her on another platform more than
    KNOCK_FALL_DY lower; one that leaves her on hers isn't one."""
    b = plat(model, 360, 300)
    w = world(model, b.id, 385.0, monsters=[("Snail", b.id, 360.0)])
    w.monsters[0].speed = 0.0
    w.advance(2.0)
    assert w.touches >= 1 and w.plat is not None and w.plat.id != b.id
    assert len(w.knock_log) == 1 and w.knock_log[0][1:3] == (b.id, w.plat.id) and w.knock_log[0][3] > 100
    out = harness.outcomes(w, [], 60.0)
    assert out["knock_falls"] == 1 and out["knock_fall_dy"] == w.knock_log[0][3]
    c = plat(model, 500, 400)
    w = world(model, c.id, 385.0, monsters=[("Snail", c.id, 360.0)])
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


def test_hp_lost_on_the_trips_to_rest_is_counted_between_their_events(model):
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0))
    w.touch_log = [(5.0, 0.014), (12.0, 0.02), (13.0, 0.0), (30.0, 0.01)]  # (t, the HP it took: 0 evaded)
    t0 = 1000.0
    records = [{"topic": "session", "t": t0}, {"topic": "event", "kind": "rest_trip", "t": t0 + 10.0},
               {"topic": "event", "kind": "rest", "t": t0 + 20.0}, {"topic": "event", "kind": "rest_trip", "t": t0 + 25.0}]
    out = harness.outcomes(w, records, 60.0)
    assert out["rest_trips"] == 2  # the last one still going at the end
    assert out["rest_hp_lost"] == pytest.approx(0.03)  # at 12 s and 30 s; 13 s was evaded
    assert out["hp_lost"] == pytest.approx(0.044)
    base = [dict(_result(k, 50.0), rest_hp_lost=0.05, rest_trips=2) for k in range(1, 11)]
    assert harness.compare_rest(base, [dict(r, rest_hp_lost=0.02) for r in base])[0]
    assert not harness.compare_rest(base, [dict(r, rest_hp_lost=0.08) for r in base])[0]
    # Runs saved before the metric existed, or with no seed in common, never pass.
    assert not harness.compare_rest([_result(k, 50.0) for k in range(1, 11)], base)[0]
    assert not harness.compare([], [])[0] and not harness.compare(base, [_result(k, 60.0) for k in range(11, 21)])[0]
    assert not harness.compare_rest(base, [dict(r, seed=r["seed"] + 10) for r in base])[0]
    # No trip to rest on a side: nothing measured, nothing passed.
    assert not harness.compare_rest(base, [dict(r, rest_hp_lost=0.0, rest_trips=0) for r in base])[0]
    # The bar is over the seeds both have: a 20-seed base against 10 seeds of a candidate compares those 10.
    harder = base + [dict(_result(k, 50.0, share=0.1, hp=1.0)) for k in range(11, 21)]
    assert not harness.compare(harder, [_result(k, 60.0, share=0.4, hp=0.2) for k in range(1, 11)])[0]
    # Fewer than 10 seeds in common never pass; a death counts on any seed, compared or not.
    better = [dict(r, exp_per_min=60.0, rest_hp_lost=0.01) for r in base]
    assert not harness.compare(base[:9], better[:9])[0] and not harness.compare_rest(base[:9], better[:9])[0]
    assert harness.compare(base, better)[0] and harness.compare_rest(base, better)[0]
    died = better + [dict(better[0], seed=15, died_at=600.0)]
    assert not harness.compare(base, died)[0] and not harness.compare_rest(base, died)[0]


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


def test_her_box_can_be_the_clients_in_the_simulated_world(model):
    # Realism.her_box (`--world client-body`): 22 either side of her feet and 65 tall (motion.bodies), not the
    # simulator's 15 and 55: a slime walking on B, 140 above C's floor, never reaches her there; one on a platform 60
    # over her does with the client's box only (her head 65 up).
    from gamelab.motion import bodies
    assert world_mod.HER_BOX_CLIENT == (bodies.HER_HALF, bodies.HER_TALL) == (22.0, 65.0)
    assert harness.WORLDS["client-body"] == {"her_box": world_mod.HER_BOX_CLIENT}
    sc = harness.scenario("loop", 1, "game", 1.0, world=("client-body",))
    assert sc.realism.her_box == (22.0, 65.0) and Realism().her_box is None
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


def test_a_trip_to_rest_is_measured_both_ways():
    if not scenarios.available():
        pytest.skip("needs the real map's data")
    ok, lines = harness.rest_trials(1)
    assert lines[0].startswith("transit: HP lost") and lines[1].startswith("avoid: HP lost") and "rest bar" in lines[2]
    assert harness.rest_trip(True, 0)["kills"] == 0  # keeping clear, she shoots nothing


def test_a_short_harness_run_on_the_real_map():
    if not scenarios.available():
        pytest.skip("needs the real map's data")
    from gamelab.motion import route
    out = harness.run_one("tour", 1, "game", minutes=1.0)
    assert out["policy"] == "tour" and "SimTimeout" in out["result"] and out["exp_per_min"] >= 0
    assert set(harness.COLUMNS) - {"seed"} <= set(out) and out["ways"] == route.ROUTES
    ways = harness.run_one("tour", 1, "game", minutes=1.0, ways=3)  # (--ways: route.ROUTES for the run, put back)
    assert ways["ways"] == 3 and route.ROUTES == out["ways"]


def test_a_harness_run_takes_her_looks_from_the_feed_with_todays_look_costs():
    if not scenarios.available():
        pytest.skip("needs the real map's data")
    from gamelab.sim import fakes
    coded = fakes.LOOK_S
    out = harness.run_one("loop", 1, "game", minutes=0.5, costs="measured")
    assert out["feed"] and out["costs"] == "measured" and fakes.LOOK_S == coded  # put back after the run
    rec = out["feed_record"]
    assert rec["frames"] > 0 and rec["senses"]["looks"] > 0 and out["looks"] == rec["senses"]["looks"]
    assert out["nav"]["glances"] == rec["senses"]["glances"] > 0


def test_the_feeds_costs_as_measured_at_1366x768_are_the_runs_alone(monkeypatch):
    # --costs measured-1366: the fakes' look costs as `measured`'s, and the feed made for the run takes
    # feed.COSTS_1366 (SyncFeed's costs are COSTS as it is when it's made); both put back after the run.
    if not scenarios.available():
        pytest.skip("needs the real map's data")
    from gamelab.localization import feed
    from gamelab.sim import fakes
    seen = []
    monkeypatch.setattr(scenarios, "run_sim", lambda sc: seen.append((feed.COSTS, fakes.LOOK_S)) or 1 / 0)
    coded, look = feed.COSTS, fakes.LOOK_S
    with pytest.raises(ZeroDivisionError):
        harness.run_one("loop", 1, "game", minutes=0.5, costs="measured-1366")
    assert seen == [(feed.COSTS_1366, harness.COSTS["measured"]["LOOK_S"])]
    assert feed.COSTS is coded and fakes.LOOK_S == look
    assert feed.COSTS_1366.refind == feed.COSTS.refind and feed.COSTS_1366.wide < feed.COSTS.wide
    made = feed.SyncFeed.__new__(feed.SyncFeed)
    monkeypatch.setattr(feed, "COSTS", feed.COSTS_1366)
    made.__init__(type("Step", (), {"senses": None})(), lambda ropes: None)
    assert made.costs is feed.COSTS_1366 and made.searches.costs is feed.COSTS_1366


def test_the_stacks_computing_is_timed_between_its_calls_and_the_run_is_as_it_was():
    """sim.latency times the stack's own computing between two calls to the game, by the call that ended
    it, without charging it to the fake clock: the run is the harness's own."""
    from gamelab.sim import latency
    if not scenarios.available():
        pytest.skip("needs the real map's data")
    found, (timed,) = latency.measure("loop", [1], 0.5)
    assert timed == harness.run_one("loop", 1, "game", minutes=0.5)
    assert found.times["send"] and found.times["sleep"] and min(min(xs) for xs in found.times.values()) >= 0
    assert fakes.Clock.charge is Clock.charge and Clock.charge.__module__ == "gamelab.sim.fakes"  # put back
    lines = found.table()
    assert lines[0].startswith("before") and lines[-1].startswith("any call")
    assert sum(int(line.split()[-6]) for line in lines[1:-1]) == int(lines[-1].split()[-6])


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
    assert harness.navigation(log, model) == {"cut_hit": 1, "cut_off_path": 1, "replans": 5, "misses": 3, "falls": 2}


def test_a_world_drops_what_its_maps_monsters_drop(model):
    """The harness's `game-drops` (Realism.drops MAP_DROPS): on the first map what was seen there in game (GAME_DROPS);
    on another a monster seen in game drops as it did there, and its other drops (perception.drops) at OSMLib's
    reference rates; one not seen in game its drops at those rates (an item with none as its rarest), and mesos."""
    from types import SimpleNamespace
    from gamelab.perception.drops import FIRST_MAP, MONSTERS
    from gamelab.sim.world import GAME_DROPS, MAP_DROPS, MESOS_SHARE, NO_RATE, game_drops

    def a_map(map_id, *names):
        return SimpleNamespace(id=map_id, spawns=[SimpleNamespace(name=n) for n in names])

    assert game_drops(a_map(FIRST_MAP, "Pig", "Slime")) == GAME_DROPS
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
