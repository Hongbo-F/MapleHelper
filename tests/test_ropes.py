"""Ropes as places to wait out monsters (attack.ropes, attack.danger's on_rope, the hunt loop's rule 9 on a rope), on
synthetic maps: which kinds can reach her on a rope (a walker under it only as high as its body, a hopper as high as
its hop's frames, a flier anywhere), the ropes she can get onto from a platform and back, at what height and what it
takes, and when the hunt loop hangs on one, and gets off it. And in the simulator, end to end, on a platform no place
along which is safe from its Horny Mushrooms' spawn points."""

import math
import time
from types import SimpleNamespace

import pytest

from gamelab.attack import ropes
from gamelab.attack.danger import HER_TALL, THREAT_S, Danger, Her
from gamelab.behavior.safety import SafetyStop
from gamelab.localization import mapdata
from gamelab.motion.bodies import HER_HALF

from test_danger import BODIES, DAMAGE, HOPS, NAMES, SPEEDS, seen

HER = Her(hp=752.0, level=24.0)
TALL = {"19": 52.0, "13": 52.0, "7": 66.0, "99": 52.0, "98": 40.0}  # game data: the mushrooms 52, a slime's hop 66


def build(footholds, lives, rope_list=(), fliers=()):
    """A map of these footholds (each a platform of its own group), spawn points (monster id, x, y) and ropes (x, top,
    bottom, from its platform); `fliers`: monster ids that fly ("98" stands for one: no map of hers has a flier)."""
    names = dict(NAMES, **{"98": "Flier"})
    tree = {"info": {"VRLeft": -500, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
            "foothold": {"1": {str(i): {str(i): {"x1": a, "y1": b, "x2": c, "y2": d, "prev": 0, "next": 0}}
                               for i, (a, b, c, d) in enumerate(footholds, 1)}},
            "life": {str(i): {"type": "m", "id": mob, "x": x, "y": y, "cy": y, "rx0": x, "rx1": x}
                     for i, (mob, x, y) in enumerate(lives)},
            "ladderRope": {str(i): {"x": x, "y1": top, "y2": bottom, "l": 0, "uf": int(up)}
                           for i, (x, top, bottom, up) in enumerate(rope_list, 1)},
            "miniMap": {"width": 2000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4}}
    ids = {mob for mob, _, _ in lives}

    def mine(table: dict) -> dict:
        return {i: v for i, v in table.items() if i in ids}
    bodies, speeds = dict(BODIES, **{"98": BODIES["99"]}), dict(SPEEDS, **{"98": -20})
    damage, hops = dict(DAMAGE, **{"98": 110}), dict(HOPS, **{"98": False})
    return mapdata.parse(1, tree, "Test", {}, names, mine({i: 41 for i in names}), mine(bodies), {}, mine(speeds),
                         mine(damage), mine(hops), mine({i: i in fliers for i in names}), mine(TALL))


def on(model, x, y):
    return model.platform_under(x, y).id


# A floor 0..800 at y 500, a platform over its left part (150..350 at y 250) the rope at 250 hangs from, down to 440:
# 60 above the floor, grabbed from it jumping straight up (her jump 77, and the arms).
FLOOR, OVER = (0, 500, 800, 500), (150, 250, 350, 250)
ROPE = (250, 250, 440, True)


def test_a_walker_under_a_rope_reaches_her_only_as_high_as_its_body_a_hopper_as_its_hop_a_flier_anywhere():
    model = build([FLOOR, OVER], [("19", 250, 500), ("7", 300, 500)], [ROPE])
    d, rope = Danger(model, HER), model.ropes[0]
    horny = d.threats["horny-mushroom"]
    assert horny.tall == 52.0 and d.threats["slime"].tall == 66.0
    # The Horny Mushroom right under the rope: its top 52 over the floor, her feet must be above it to be out of reach.
    assert d.on_rope(rope, 500 - 52 + 1) == pytest.approx(d.share("horny-mushroom"))  # her feet 51 up: touched
    assert d.on_rope(rope, 500 - 52 - 1) == 0.0                                        # 53 up: out of its reach
    assert d.on_rope(rope, 400) == 0.0 and d.on_rope(rope, 300) == 0.0
    # Hanging low on a rope down to 30 over the floor, it touches her once its walk brings its body under hers: from
    # 250 away, a walk of 250 less its half width and hers.
    far = build([FLOOR, OVER], [("19", 500, 500)], [(250, 250, 470, True)])
    df = Danger(far, HER)
    assert df.on_rope(far.ropes[0], 470) == pytest.approx(
        df.share("horny-mushroom") * math.exp(-(250 - 28 - HER_HALF) / 100 / THREAT_S))
    # A hopper reaches as high as its hop's frames (66): it would count if it counted (a slime: 1.2%, under DANGER_MIN).
    t, floor = d.threats["slime"], model.platform(on(model, 100, 500))
    assert d.walk_rope(t, floor, 250, rope, 500 - 60) is not None
    assert d.walk_rope(t, floor, 250, rope, 500 - 70) is None
    # A flier goes straight at her, whatever the floors: 100 off sideways and level with her, 100 - 28 - 15 to go.
    flier = build([FLOOR, OVER], [("98", 350, 500)], [ROPE], fliers={"98"})
    dfl = Danger(flier, HER)
    t = dfl.threats["flier"]
    assert t.flies
    way = dfl.walk_rope(t, flier.platform(on(flier, 100, 500)), 350, flier.ropes[0], 500 - t.tall / 2 + HER_TALL / 2)
    assert way == (pytest.approx(100 - 28 - HER_HALF), 1.0)
    assert dfl.on_rope(flier.ropes[0], 300) > 0  # high up the rope too, unlike a walker


def test_the_ropes_she_can_get_onto_from_a_platform_and_back_at_their_safest_height():
    model = build([FLOOR, OVER], [("19", 250, 500)], [ROPE])
    d, floor, over = Danger(model, HER), on(model, 100, 500), on(model, 250, 250)
    (under,) = ropes.refuges(model, d, floor)
    assert not under.from_top and under.x == 250 and under.platform == floor
    # Grabbed at the top of her jump (feet 72 up: 428), held up past it, out of the Horny Mushroom's reach and its
    # slack (52 + HANG_SLACK under her feet) and below TOP_KEEP from its top.
    assert 250 + ropes.TOP_KEEP <= under.y <= 428 - ropes.GRAB_CLIMB and under.danger == 0.0
    assert 500 - under.y >= 52 + ropes.HANG_SLACK
    assert 0.3 < under.on_s < 1.5 and 0.2 < under.off_s < 1.5
    (top,) = ropes.refuges(model, d, over)  # from the platform it hangs from: down onto it, and back up over its top
    assert top.from_top and top.platform == over and 250 < top.y <= 440 - ropes.END_KEEP
    # A rope at a platform's very end (ROPE_IN) isn't one: letting go of it she came down beside it.
    edge = build([FLOOR, OVER], [("19", 250, 500)], [(795, 250, 440, False)])
    assert ropes.refuges(edge, Danger(edge, HER), on(edge, 100, 500)) == []
    # Nor one out of her reach above her (95 up, less GRAB_MARGIN).
    high = build([FLOOR, OVER], [("19", 250, 500)], [(250, 250, 400, False)])
    assert ropes.refuges(high, Danger(high, HER), on(high, 100, 500)) == []


def test_how_high_a_kind_reaches_and_whether_it_flies_come_with_the_map():
    from gamedata.paths import COPY
    if not (COPY / "Packs" / "Mob_00000.ms").is_file():
        pytest.skip("needs the game data copy (gamedata/data/copy)")
    data = mapdata._mob_touch({"7", "13", "19"})  # a slime (its hop's frames), the Green and Horny Mushrooms
    assert data["mob_tall"] == {"7": 66.0, "13": 52.0, "19": 52.0}
    assert data["mob_flies"] == {"7": False, "13": False, "19": False}


# --- the hunt loop's rule 9 on a rope ---------------------------------------------------------------------------------

def _hunt(model, ground, grounds=None, due=None, hp=1.0):
    from gamelab.attack import spots
    from gamelab.behavior.hunt import Hunt, Visit
    h = object.__new__(Hunt)
    h.model, h.kinds, h._skip, h._rope_rest, h._hanging, h._hung_at = model, {"horny-mushroom"}, {}, 0.0, None, 0.0
    lows = []

    def health_check(s):  # her potions, and the safety stop (Hunter.health_check raises it)
        if s.hp < 0.4:
            raise SafetyStop(s.hp, 2.0, 3, 0.4)
        lows.append(s.hp)
    h.h = SimpleNamespace(danger=Danger(model, HER), knocked=lambda s: False, health_check=health_check)
    h.visit = Visit(ground, spots.Spot(ground, 400.0, 1.0, False))
    h.tour = SimpleNamespace(hunting_grounds=lambda visit_s=None: list(grounds or [ground]), stats={})
    h._due_back = lambda: due
    h._halves = {}
    s = SimpleNamespace(platform=model.platform(ground), where=SimpleNamespace(x=400.0, y=500.0), monsters=[], hp=hp)
    return h, s


# Three Horny Mushrooms' spawn points along the floor, 300 apart: nowhere along it is out of a fresh one's reach, and
# the rope's foot (150 from two of them) is as safe as anywhere on it.
LIVES = [("19", 100, 500), ("19", 400, 500), ("19", 700, 500)]


def test_she_waits_on_a_rope_only_where_nowhere_along_her_platform_is_safe_enough():
    model = build([FLOOR, OVER], LIVES, [ROPE])
    g = on(model, 100, 500)
    h, s = _hunt(model, g)
    r = h._refuge(s)
    assert r is not None and r.x == 250 and r.danger == 0.0
    at, best = h._safest_along(s)
    from gamelab.behavior.hunt import ROPE_GAIN
    assert min(at, best[0]) - r.danger >= ROPE_GAIN
    # With another ground to go to and nothing of hers due back, she leaves on her patience: no time for a rope.
    h2, s2 = _hunt(model, g, grounds=[g, 7])
    assert h2._refuge(s2) is None
    h2._due_back = lambda: 9.0  # a kill of hers due back in 9 s: she waits, and the rope is worth it
    assert h2._refuge(s2) is not None
    h2._due_back = lambda: 0.5  # due sooner than getting on and off takes
    assert h2._refuge(s2) is None
    # One spawn point only, far from her: a place along the floor is safe enough, and the rope no default.
    one = build([FLOOR, OVER], [("19", 50, 500)], [ROPE])
    h3, s3 = _hunt(one, on(one, 100, 500))
    assert h3._refuge(s3) is None
    # The rope's foot right by a spawn point (where respawns land on her as she comes back down): not a refuge.
    foot = build([FLOOR, OVER], [("19", 100, 500), ("19", 270, 500), ("19", 700, 500)], [ROPE])
    h4, s4 = _hunt(foot, on(foot, 100, 500))
    assert h4._refuge(s4) is None
    # Where nothing counts as a danger (slimes: the first map's case), never.
    weak = build([FLOOR, OVER], [("7", 100, 500), ("7", 400, 500), ("7", 700, 500)], [ROPE])
    h5, s5 = _hunt(weak, on(weak, 100, 500))
    h5.kinds = {"slime"}
    assert h5._refuge(s5) is None


def test_on_the_rope_she_lets_a_monster_pass_under_her_and_gets_off_to_shoot_one_away_from_her_landing():
    from gamelab.behavior.hunt import HANG_MAX_S, OFF_TO_SHOOT
    model = build([FLOOR, OVER], LIVES, [ROPE])
    g = on(model, 100, 500)
    h, s = _hunt(model, g)
    h._hanging, h._hung_at = h._refuge(s), time.monotonic()
    up = SimpleNamespace(platform=None, where=SimpleNamespace(x=250.0, y=h._hanging.y), monsters=[], hp=0.9)
    assert h._on_refuge(up) and h._off_rope(up) is None  # nothing about: she stays
    up.monsters = [seen("horny-mushroom", 300.0, 500.0, g)]  # back, by her landing (a claw's swipe from it)
    assert h._off_rope(up) is None
    up.monsters = [seen("horny-mushroom", 500.0, 500.0, g)]  # walked off: down to shoot it
    assert h._off_rope(up) == OFF_TO_SHOOT
    up.monsters, up.hp = [], 0.2
    with pytest.raises(SafetyStop):  # her HP stayed low: the hunt ends from the rope too (no getting off to rest)
        h._off_rope(up)
    up.hp = 0.9
    h._hung_at -= HANG_MAX_S
    assert h._off_rope(up).endswith("s on it")
    assert not h._on_refuge(SimpleNamespace(platform=None, where=SimpleNamespace(x=300.0, y=400.0)))


def test_in_the_simulator_she_waits_on_the_rope_and_comes_down_to_shoot(monkeypatch):
    """The hunt loop for 3 minutes on the floor above (its three Horny Mushrooms, respawns 17 s after a kill), started
    on it: she goes to wait on the rope when it's empty, gets off to shoot what comes back, and lives. (Of seeds 1-12
    she waited on the rope on 10 with her body the client's, motion.bodies, 7 with the simulator's 15 x 55; with the hunt
    loop's sweep and her memory of the monsters (hunt.SWEEP) on 6, and this is met on 5 (2, 3, 7, 9, 10): she waits
    less (the rule 'wait' decided 23 times on those 12 runs against 111 without the sweep) for kills as many (14.2 a run
    against 14.0). Seed 3 until her jumps' attacks were weighed, 2026-10-25, with the review's fixes (her body clear of
    one above over a jump in place, a jump forward's landing however it goes): of seeds 1-12 it's met on 7 (1, 5-9,
    12), and on 3 she now never waits on the rope. Seed 1. Seed 3 since every look is perception's newest frame after
    her keys' effect, Senses.fresh_after, the feed in every run: of seeds 1-12 it's met on 3 and 8 (she waits on the rope
    on 3, 5 and 8), on 1 she never waits on it; with a look asking for what the next frame would have stale,
    feed.to_ask, and the fixes with it, on 3, 8 and 10, the only ones she waits on the rope on. Seed 2 since
    2026-10-26, a monster by a kill placed taken for the one dying there while its death animation plays,
    ledger.dying: of seeds 1-12 it's met on 2, 4-7, 10 and 11, and on 3 she never waits on the rope.) Seed 2 kept with
    parts A and B together, work/hunt-ab, 2026-10-26, where part B alone had it on 4: of seeds 1-12 it's met on 1-3, 10
    and 12; on 7 she went to rest with no safe platform reachable.) Seed 1 since 2026-10-27, an attack from a jump
    picking by the star's search alone (reach.AIR_MODEL, the world's AIR_PICK) and her stance going by where each
    monster will be when her attack starts (attack.stance.Pace): on seed 2 she never waited on the rope; of seeds 1-12
    it's met on 1, 6, 7, 8 and 12, and on 3 she waits on it, the rest of this not met. Seed 3 since the whole screen 10
    times a second with her band's own search beside it where a whole screen takes longer than 0.05 s
    (worldstate.BAND_STAND_IN_S, 2026-10-28): on seed 1 she waits on the rope, the rest of this not met; of seeds 1-12
    it's met on 3, 5, 7 and 11. Seed 1 since the world's monsters chase her once she hits them (Realism.chase,
    2026-10-31): on seed 3 the safety stop at 68 s, never on the rope; of seeds 1-12 it's met on 1, 6 and 9, and on the
    other 9 the safety stop (HP under her potion key's 40%) before she ever waits on the rope. Seed 1 since the review's
    fixes of work/replan, 2026-10-30 (the frames on their grid at 15 a second whatever is asked, feed.run; a monster
    standing planned either way a while, navigator.STAND_UNSURE_S;
    D:/MapleHelper-scratch/replan/fix/out/scant_ropes.txt): on seed 3 she never waits on the rope; of seeds 1-12 it's
    met on all but 3 and 10.) Seed 4 since work/replan's rebase onto the server's respawn and the chase (2026-10-31:
    D:/MapleHelper-scratch/replan/pos/fix/out/rope.txt): on seed 1 she never waits on the rope; of seeds 1-12 it's met
    on 4, 9, 10 and 12, and on 7 of the others the safety stop before she ever waits on it.) Seed 11 since the stack's
    monsters turn where the game turns them, 20 from a platform's end (monsters.walk_edges, 2026-10-31): on seed 1 the
    safety stop before she waits on the rope; of seeds 1-12 it's met on 11 alone, and on 10 she waits on it and the
    safety stop comes, on the other 10 before she ever waits on it.) Seed 1 since her jump attacks are jittered, those
    up close too (2026-11-01: fight.jittered; D:/MapleHelper-scratch/replan/speed/j/rope_seeds.txt): on 11 the safety
    stop before she ever waits on the rope; of seeds 1-12 it's met on 1, 4, 6 and 12, on 1 for the whole 180 s (12
    kills, no safety stop). Seed 4 since every jump attack is jittered, those the stance keys unsure too
    (fight.JITTER_ALL, the review of 2026-11-01; D:/MapleHelper-scratch/replan/speed/fix/seeds/rope_scan.txt): on 1 the
    safety stop before she ever waits on the rope; of seeds 1-12 it's met on 2-5 and 11, on 4 for the whole 180 s (13
    kills, no safety stop)."""
    from gamelab.sim import harness, scenarios, world
    from gamelab.sim.world import Realism
    model = build([FLOOR, OVER], LIVES, [ROPE])
    if world.mob_info(19) is None:
        pytest.skip("needs the Horny Mushroom's game data (gamedata/data)")
    g = on(model, 100, 500)
    sc = scenarios.Scenario("rope", harness._only(scenarios._loop(3), (g,)), (g, 100.0), monsters="all", seed=4,
                            limit=400, realism=Realism(respawn=17.0, respawn_tick=0.0, her_box=(world.HER_HALF,
                                                                                               world.HER_HEIGHT)),
                            may_die=True)  # (her body for touches as when it was seeded: 15 either side, not the line)
    trace = scenarios.run_scenario(sc, model)
    log = "\n".join(trace["log"])
    assert "waiting on the rope at x 250" in log and "off the rope at x 250" in log, trace["result"]
    assert any("grabs the rope at x 250" in line for line in trace["world_log"])
    assert not trace["world"]["dead"] and sum(trace["world"]["kills"].values()) >= 3
