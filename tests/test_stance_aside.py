"""The jump aside and the hop up (attack.stance._aside, _hop; the user, 2026-11-05: "no, jump aside and attack from a distance", hop up onto
the pile "Yes"): frozen looks of the failsafe's recordings, each as the replay of the 14 tours has it (her place, the look's age, her
invulnerability, each monster's walk), and the keys they go by."""

import random
from collections import Counter
from types import SimpleNamespace

import pytest
from maplehelper.engine import LEFT, RIGHT

from gamelab import telemetry
from gamelab.attack import danger, reach, stance as S
from gamelab.behavior import fight
from gamelab.control.actions import JB_RUN_MS, JB_TURN_MS, Actions
from gamelab.localization import monsters as mons
from gamelab.localization.monsters import Monster, speeds
from gamelab.motion import physics
from gamelab.sim import scenarios

REAL = pytest.mark.skipif(not scenarios.available(scenarios.TREE), reason="needs Tree Dungeon's game data (gamedata/data)")
H, G, SL = "horny-mushroom", "green-mushroom", "slime"
# (clip row): her x, feet, platform, facing, s to her next keys, s of her invulnerability left, her target's x, its monsters (kind, x, y, platform, bar, walk)
ROWS = {
    "231613 a02": (-171.5, -1594.9, 7, RIGHT, 0.069, 1.13, -220, [(H, -220, -1588.0, 7, None, 0.0), (H, 237.0, -1620.4, 6, None, 0.0)]),
    "231613 a03": (-176.5, -1595.6, 7, LEFT, 0.069, 0.0, -212.0, [(H, -212.0, -1589.8, 7, None, 0.0), (H, 269.0, -1617.1, 6, None, 0.0)]),
    "081701 a04": (254.5, 224.1, 33, RIGHT, 0.137, 0.0, 301.0, [(SL, 17.0, 232.1, 33, None, -100.0), (SL, 301.0, 223.9, 33, 0.59, -100.0)]),
    "080032 a04": (-63.5, -49.3, 29, RIGHT, 0.202, 0.68, -80.0, [(G, -10.0, -59.2, 29, None, -100.0), (G, -1.0, -2.5, 30, None, 100.0), (G, -80.0, -48.7, 29, None, 0.0)]),
    "080032 a05": (-70.5, -49.1, 29, LEFT, 0.144, 1.34, -52.0, [(G, 61.0, -4.2, 30, 0.59, 100.0), (G, -52.0, -49.7, 29, None, -100.0)]),
    "111448 a04": (123.5, -1565.4, 8, RIGHT, 0.117, 0.0, 166.0, [(G, 166.0, -1577.0, 8, 0.89, -100.0)]),
    "235621 a05": (-153.5, -1248.9, 11, RIGHT, 0.053, 0.0, -120.0, [(G, -120.0, -1238.2, 11, None, 0.0)]),
    "233732 a01": (18.5, -1110.0, 14, None, 0.052, 0.0, -119.0, [(G, 177.0, -1194.9, 12, None, 0.0), (H, -119.0, -1178.0, 13, None, -100.0), (H, -3.0, -1182.2, 13, None, 100.0), (H, -16.0, -1181.6, 13, None, 100.0)]),
    "233732 a03": (11.5, -1110.2, 14, LEFT, 0.419, 0.0, -189.0, [(G, 180.0, -1195.3, 12, None, 0.0), (H, -189.0, -1179.5, 13, None, 100.0), (H, -74.0, -1178.9, 13, None, -100.0), (H, 4.0, -1183.9, 13, None, 0.0)]),
    "233732 a04": (11.5, -1110.2, 14, LEFT, 0.416, 0.0, -160.0, [(G, 196.0, -1194.0, 12, None, 100.0), (H, 4.0, -1183.9, 13, None, 0.0), (H, -160.0, -1180.6, 13, None, -100.0), (H, -103.0, -1176.4, 13, None, 100.0)]),
    "111448 a07": (91.5, -1526.6, 9, RIGHT, 0.047, 0.0, -68.0, [(H, -68.0, -1599.2, 7, None, 100.0), (H, 6.0, -1466.7, 10, None, 100.0), (G, 133.0, -1569.0, 8, 0.89, 0.0), (H, 138.0, -1457.7, 10, None, 0.0)]),
    "111448 a08": (84.5, -1561.1, 8, LEFT, 0.046, 0.0, -145.0, [(G, 141.0, -1572.0, 8, 0.89, 0.0), (H, -145.0, -1592.9, 7, None, -100.0)]),
}


def _at(row: str, aside: bool = True, hop: bool = True, moves: int = 0, monkeypatch=None) -> tuple:
    """What her stance says at the look of clip row `row`, the target-first planner's way, with the switches as said."""
    x, y, on, facing, lead, inv, tx, rows = ROWS[row]
    model = scenarios.real_map(scenarios.TREE)
    here = model.platform(on)
    around = [Monster(SimpleNamespace(kind=k, fading=None), mx, my, p, bar, 0, "") for k, mx, my, p, bar, _ in rows]
    walks = {id(m): r[5] for m, r in zip(around, rows)}
    target = min((m for m in around if m.x == tx), key=lambda m: abs(m.x - tx))
    body = mons.bodies(model)
    tops, halves = {k: b.top for k, b in body.items()}, {k: b.half for k, b in body.items()}
    pace = S.Pace(lead, lambda m: walks[id(m)], speeds(model), halves, None, inv)
    done = []

    def jumps():
        if not done:
            done.append(reach.coverage(model, here, x, y, [target], sizes=body, others=around,
                                       where=lambda m, t: pace.at(m, x, pace.lead + t, S.walks_on(m, here)), lands=S.jump_room(here)))
        return done[0]
    monkeypatch.setenv("GAMELAB_JUMP_ASIDE", "1" if aside else "0")
    monkeypatch.setenv("GAMELAB_HOP_UP", "1" if hop else "0")
    return S.stance(x, y, here, [target], False, facing, can_jump=lambda a, b: S.open_above(model, here, a, b), tops=tops, halves=halves,
                    around=around, danger=danger.of_map(model, danger.Her(1244.0, 35.0, 117.0)), jumps=jumps, edges=S.drop_limits(model, here),
                    up=lambda v: S.overhead(model, here, v), pace=pace, limits=S.move_limits(model, here), moves=moves, among=around)


@REAL
@pytest.mark.parametrize("row, faces, lands", [
    ("231613 a02", LEFT, -60),   # (a wall's foot, a tap first and a star stopped at the trunk: away from it)
    ("231613 a03", LEFT, -65),   # (a wall's foot, facing it, in under the search: away)
    ("081701 a04", RIGHT, 144),  # (the map's end, a slime coming: away, under P32's end)
    ("080032 a04", RIGHT, -177),  # (one she walked past behind her: on, to turn back at it)
    ("080032 a05", RIGHT, -184),  # (a bare jump back, its key picking nothing: the same keys with no key)
])
def test_an_attack_the_search_says_takes_nothing_goes_as_a_jump_with_no_key(row, faces, lands, monkeypatch):
    act = _at(row, monkeypatch=monkeypatch)
    plan = S.plan_of(act)
    assert act[0] == "jump_back" and act[1] == faces and plan.why == S.ASIDE
    assert plan.key_ms == 0 and plan.keys == (0, 0)  # (nothing in the air: jitter none, no attack key)
    assert plan.way == (-1 if faces == RIGHT else 1) and plan.land_x == pytest.approx(lands, abs=2)
    assert plan.target is not None


@REAL
@pytest.mark.parametrize("row, why", [
    ("111448 a04", "the search says a star: of these 83-89% hit, 55-61% kill"),
    ("235621 a05", "the short way over walks into it (MEET_PRICE_S), the wall's behind her"),
])
def test_where_the_jump_is_the_dearer_the_attack_goes(row, why, monkeypatch):
    assert S.plan_of(_at(row, monkeypatch=monkeypatch)) is None or S.plan_of(_at(row, monkeypatch=monkeypatch)).why != S.ASIDE, why
    assert _at(row, monkeypatch=monkeypatch)[0] == "jump_throw"


@REAL
def test_one_jump_aside_then_a_throw_whatever_she_finds_where_she_lands(monkeypatch):
    assert S.plan_of(_at("231613 a03", moves=1, monkeypatch=monkeypatch)) is None  # (it's a move: fight.keyless, st.moves)
    assert _at("231613 a03", moves=1, monkeypatch=monkeypatch)[0] == "jump_throw"


@REAL
@pytest.mark.parametrize("row", ["231613 a03", "081701 a04", "080032 a04", "080032 a05", "233732 a03", "111448 a07"])
def test_off_she_does_as_before(row, monkeypatch):
    plan = S.plan_of(_at(row, False, False, monkeypatch=monkeypatch))
    assert plan is None or plan.why not in (S.ASIDE, S.HOP)


@REAL
@pytest.mark.parametrize("row, kind, key, onto", [
    ("233732 a03", "up", reach.TOP_KEY_MS, 13),  # (under P13: a level star under it, the window shut by her pick)
    ("233732 a04", "forward", reach.TOP_KEY_MS, 13),
    ("111448 a07", "up", physics.AIR_TURN_ATTACK_MS, 8),
])
def test_a_throw_from_the_ground_that_takes_nothing_goes_up_onto_the_platform_above(row, kind, key, onto, monkeypatch):
    act = _at(row, monkeypatch=monkeypatch)
    plan = S.plan_of(act)
    assert act[0] == "jump_throw" and plan.why == S.HOP and plan.key_ms == key
    assert plan.shot.kind == kind and plan.shot.land.id == onto and plan.shot.lands_sure and plan.keys == plan.shot.window


@REAL
def test_a_jump_shot_landing_back_comes_before_the_hop_and_the_throw_stays_where_the_search_picks(monkeypatch):
    plan = S.plan_of(_at("111448 a08", monkeypatch=monkeypatch))
    assert plan.why == "above" and plan.shot.land.id == 8  # (the stance's own _jump_shot, landing her back)
    assert _at("233732 a01", monkeypatch=monkeypatch)[0] == "throw"  # (a star at the look: 82% hit, 61% kill)


@REAL
def test_the_magicians_answers_have_neither(monkeypatch):
    monkeypatch.setattr(S, "POINT_BLANK", False)
    for row in ("231613 a03", "233732 a03"):
        plan = S.plan_of(_at(row, monkeypatch=monkeypatch))
        assert plan is None or plan.why not in (S.ASIDE, S.HOP)


def _actions(sent: list) -> Actions:
    key = lambda name: SimpleNamespace(name=name)
    a = Actions(SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), attack=key("f"), lucky_seven=key("shift"), pick_up=key("z"),
                                hp_potion=None, chair=None), skill=True)
    a.play = lambda events, name: sent.append((name, events))
    return a


def test_her_jump_asides_keys_are_her_jump_backs_with_no_attack():
    sent, told = [], []
    a = _actions(sent)
    a.throw_listeners.append(lambda when, side: told.append(side))
    assert a.jump_back(RIGHT, 0) is True and a.keyless  # (a move, no throw: fight.keyless)
    (name, events), = sent
    assert name == "jump_back" and not told and a.facing == RIGHT
    at = lambda k, how: next(t for t, n, h_ in events if n == k and h_ == how)
    assert at("left", "down") == 0 and at("alt", "down") == JB_RUN_MS  # (her run-up: live 62 times, a median 95 units)
    assert at("left", "up") == at("right", "down") == JB_TURN_MS  # (she turns in the air, to face what she left)
    assert {n for _, n, _ in events} == {"left", "right", "alt"}  # (no attack key)
    assert a.lock_until == pytest.approx(a.sent_at + JB_RUN_MS / 1000 + physics.JUMP_LAG + physics.AIR_S, abs=0.01)  # (her next keys as she's down)


def test_a_keyed_jump_back_still_keys():
    sent = []
    a = _actions(sent)
    assert a.jump_back(RIGHT, 120) is True and not a.keyless
    assert "shift" in {n for _, n, _ in sent[0][1]}


@REAL
def test_her_jump_aside_is_a_move_and_its_record_says_where_she_should_land(monkeypatch):
    act = _at("231613 a03", monkeypatch=monkeypatch)
    logged = []
    h = SimpleNamespace(log=logged.append, jump_counts=Counter(), actions=SimpleNamespace(faces=lambda side: True), jitter=random.Random(1))
    s = SimpleNamespace(where=SimpleNamespace(x=-176.5, y=-1595.6), platform=SimpleNamespace(id=7), monsters=[])
    keyed = fight.jittered(h, act, s)
    assert S.plan_of(keyed).key_ms == 0  # (nothing to jitter)
    with telemetry.captured() as records:
        fight.log_jump(h, s, keyed)
    jump = next(r for r in records if r.get("topic") == "jump")
    assert (jump["kind"], jump["jump"], jump["key_ms"], jump["way"], jump["land_x"]) == (S.ASIDE, "back", 0, 1, S.plan_of(act).land_x)
    assert h.jump_counts["aside: back"] == 1 and "no attack, to land at x -65" in logged[0]
    assert S.act_text(act, [], -176.5, []) == "jumping aside, to land at x -65 facing left"
