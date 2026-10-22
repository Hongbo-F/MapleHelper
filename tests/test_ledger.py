"""The hunt's bookkeeping (behavior.ledger): every EXP gain a kill, placed where a star of hers explains it,
and every pile she knows of, kept from every look and every throw, whoever makes them. Nothing decides
from them yet (step 4 of docs/hunt-modes.md), so they're checked against the simulated world."""

import numpy as np
import pytest

from gamelab.behavior.ledger import GONE_KEEP_S, DropStore, Kill, Ledger
from gamelab.behavior.looting import EMPTY_LOOKS, SETTLE
from gamelab.localization import mapdata
from gamelab.localization.scene import Loot, Scene
from gamelab.perception.loot import Item
from gamelab.sim import scenarios

from test_sim import plat, tree


@pytest.fixture(scope="module")
def model():
    return mapdata.parse(1, tree(), "Test Map", {}, {"2": "Snail"}, {"2": 2})


def _shoot_a_snail(sim):
    """She throws at the snail on her platform until her EXP goes up, then looks for its drops."""
    h = sim.hunter()
    h.actions.facing = None
    for _ in range(4):
        h.look()
        h.actions.throw("right")
    for _ in range(3):
        h.look()
    h.look(loot=True)
    return {"kills": [(k.kind, k.platform) for k in h.ledger.kills], "piles": len(h.drops.known),
            "hers": [p.hers for p in h.drops.known], "status": [p.status for p in h.drops.known]}


def test_a_kill_is_placed_where_her_star_went_and_its_drops_are_expected_there(model):
    b = plat(model, 360, 300)  # snails spawn on B, so a snail seen there is placed on it
    sc = scenarios.Scenario("ledger", _shoot_a_snail, (b.id, 240.0), monsters=[("Snail", b.id, 380.0)], limit=60)
    tr = scenarios.run_scenario(sc, model)
    assert tr["world"]["kills"] == {"snail": 1}
    assert tr["result"]["kills"] == [["snail", b.id]]
    assert tr["result"]["hers"] == [True] and tr["result"]["status"] == ["seen"]  # its drops showed up


class _Hunter:
    exp_gains = 0
    named: list = []  # the kinds her gains named, the latest last (None: not known)

    def kinds_told(self, n):
        return ([None] * n + list(self.named))[-n:] if n > 0 else []


def _look(t, x=100.0, platform=None, loot=(), looked=()):
    s = Scene(t=t, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0)
    s.loot, s.looked = list(loot), list(looked)
    return s


def test_a_gain_no_throw_explains_counts_without_a_place_or_drops():
    h, store = _Hunter(), DropStore()
    ledger = Ledger(h, store)
    h.exp_gains = 1  # a kill seen only after a look without monsters, say: she threw nothing
    ledger.on_look(_look(10.0))
    (k,) = ledger.kills
    assert not k.placed and ledger.placed == [] and store.piles == []


def test_a_gain_seen_long_after_its_star_is_placed_by_a_throw_before_the_look_before():
    """A fight's last star can land after its last look, and the sweep after it takes no look for seconds:
    the gain came after the look before the one that saw it, so a throw up to STAR_WINDOW before that one
    explains it; but a throw in the last STAR_WINDOW s comes first."""
    from gamelab.behavior.ledger import STAR_WINDOW, _Throw
    from gamelab.localization.monsters import Monster
    from gamelab.perception.mobs import Mob

    def seen(kind, x):
        return Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, 500.0, 3)

    h = _Hunter()
    ledger = Ledger(h)
    ledger._throws.append(_Throw(10.0, "right", [seen("pig", 400.0)]))
    ledger.on_look(_look(10.5))                       # the fight's last look: its star still flying
    h.exp_gains = 1
    ledger.on_look(_look(10.5 + 2 * STAR_WINDOW))     # the next, after the sweep: the gain
    assert [(k.kind, k.x) for k in ledger.kills] == [("pig", 400.0)]

    h, pig = _Hunter(), seen("pig", 300.0)
    ledger = Ledger(h)
    ledger._throws.append(_Throw(9.0, "right", [seen("snail", 600.0)]))
    ledger.on_look(_look(11.2))
    ledger._throws.append(_Throw(11.4, "right", [pig]))
    h.exp_gains = 1
    s = _look(11.6)
    s.monsters, s.searched = [seen("pig", 305.0)], True  # the snail is gone from view, the pig isn't
    ledger.on_look(s)
    assert [k.kind for k in ledger.kills] == ["pig"]  # the recent throw's, though the old one's is gone


def test_a_pile_is_forgotten_only_after_two_empty_looks_apart_and_settled():
    store = DropStore()
    store.on_kill(Kill(0.0, "pig", 3, 500.0, 300.0, 3))
    view = [(3, 300.0, 700.0)]
    store.on_look(_look(1.0, looked=view))                     # still landing: not empty yet
    assert store.piles[0].empty_since is None
    store.on_look(_look(SETTLE + 0.1, looked=view))            # empty once
    store.on_look(_look(SETTLE + 0.1 + EMPTY_LOOKS / 2, looked=view))
    assert store.known                                         # not twice far enough apart
    store.on_look(_look(SETTLE + 0.2 + EMPTY_LOOKS, looked=view))
    assert store.known == []
    item = Item("mesos", 0, 0, 13, 13, 0.9)
    store.on_look(_look(20.0, loot=[Loot(item, 900.0, 300.0, 3)], looked=[(3, 800.0, 1000.0)]))
    (p,) = store.known
    assert not p.hers and p.mesos and p.status == "seen"       # drops no kill of hers explains


def test_with_standing_only_a_look_from_mid_jump_doesnt_see_a_pile_empty(monkeypatch):
    # In mid-jump a look's loot strips sit above the ground (her y is put on the platform under her): it finds no
    # drops, and two such looks forgot piles with drops still on them. With looting.STANDING_ONLY only a look that
    # shows ground under her feet can see one empty.
    from gamelab.behavior import looting
    monkeypatch.setattr(looting, "STANDING_ONLY", True)
    store = DropStore()
    store.on_kill(Kill(0.0, "pig", 3, 500.0, 300.0, 3))
    view = [(3, 300.0, 700.0)]

    def look(t, ground):
        s = _look(t, looked=view)
        s.ground = ground
        store.on_look(s)
    look(SETTLE + 0.1, (100, 100))                   # no ground under her feet: in the air
    look(SETTLE + 0.2 + EMPTY_LOOKS, (100, 100))
    assert store.known and store.piles[0].empty_since is None
    look(SETTLE + 3.0, (60, 140))                    # standing
    look(SETTLE + 3.1 + EMPTY_LOOKS, (60, 140))
    assert store.known == []


def test_gone_piles_are_dropped_a_while_later_and_the_ones_she_went_for_still_count():
    """The store keeps a gone pile GONE_KEEP_S (kept for ever, every kill added one that every look went
    through); the hunt loop's count of the piles she went for that were seen gone (its progress) stays."""
    store, view = DropStore(), [(3, 300.0, 700.0)]
    store.on_kill(Kill(0.0, "pig", 3, 500.0, 300.0, 3))
    store.on_kill(Kill(0.0, "pig", 3, 600.0, 300.0, 3))
    a, b = store.piles
    store.tried(a)                                             # she went for a, not (yet) for b
    t = SETTLE + 0.1
    store.on_look(_look(t, looked=view))
    store.on_look(_look(t + EMPTY_LOOKS + 0.1, looked=view))   # both seen empty twice: gone
    assert store.known == [] and store.piles == [a, b] and store.gone_tried == 1
    store.tried(b)                                             # a trip to b that ends as it's seen gone
    assert store.gone_tried == 2 and b.tries == 1
    store.on_look(_look(t + EMPTY_LOOKS + 0.1 + GONE_KEEP_S / 2, looked=view))
    assert store.piles == [a, b]
    store.on_look(_look(t + EMPTY_LOOKS + 0.2 + GONE_KEEP_S, looked=view))
    assert store.piles == [] and store.gone_tried == 2
    store.tried(a)                                             # tried again after it was dropped: counted once
    assert store.gone_tried == 2 and a.tries == 2


@pytest.mark.parametrize("name", ["hunt_clear_basin", "tour_hunt_basin", "nav_fight_basin"])
def test_the_ledger_counts_every_kill_and_places_most_of_them_right(name):
    if scenarios.real_map() is None:
        pytest.skip("needs the real map's data")
    made = []
    real = scenarios.Sim.hunter

    def hunter(self, **kw):
        made.append(real(self, **kw))
        return made[-1]
    scenarios.Sim.hunter = hunter
    try:
        trace, sim = scenarios.run_sim(scenarios.SCENARIOS[name])
    finally:
        scenarios.Sim.hunter = real
    world = [k for k in sim.world.kill_log if k[3] == "her"]
    kills = made[0].ledger.kills
    assert len(kills) == len(world)  # the simulator never shows two kills between looks
    placed = [k for k in kills if k.placed]
    used, right = set(), 0
    for k in placed:
        t = k.t - sim.clock.start
        i = next((i for i, (wt, kind, pid, _) in enumerate(world)
                  if i not in used and kind == k.kind and pid == k.platform and 0 <= t - wt <= 2.5), None)
        if i is not None:
            used.add(i)
            right += 1
    # Placed: over 8 seeds of each basin scenario, 0.82-1.0 of a run's kills (0.85-1.0 with her walk and
    # jump from motion.body; pooled, 0.90-0.99 before and 0.91-0.99 after). The unplaced are kills whose
    # EXP shows only after a loot sweep, more than STAR_WINDOW after the throw. Placed right: over seeds 1-8
    # of each, 0.59-0.91 of the placed (pooled 0.72-0.78), and 0.67-0.83 (0.73-0.76) once jump-backs left at
    # full speed; tour_hunt_basin's seed 22 then placed 0.57 right. The bars were 0.9 and 0.65, set on the
    # pinned seeds, and lowered twice as seeds moved; over seeds 1-16 of each, at the physics branch's base
    # (c26a177) and on that branch: placed 0.82-1.0 and 0.86-1.0 (pooled 0.89-0.97 and 0.92-0.98), placed
    # right 0.64-0.92 and 0.54-0.91 (pooled 0.74-0.76 and 0.73-0.75). The pooled shares didn't move; one seed
    # in 16 runs to either side, and the old bars failed on some seeds at the base too. With the attack
    # track's decisions (kind and platform, within 2.5 s of the world's kill): 72-78% on average over seeds
    # 1-10 of each of these scenarios, 54-87% seed by seed (61-86% before the claw's edge went by the
    # monster's size, with the same averages); the basin's pigs and slimes crowd her, walk through her and
    # hide when hit. With both tracks, over the pinned seed and seeds 1-10 of each (33 runs): placed
    # 0.85-1.0, placed right 0.62-0.90 (0.76, 0.86 and 0.83 on the pinned seeds). So the bar placed right is
    # the attack track's 0.6, not the physics track's 0.55; placed stays at the physics track's 0.8 (0.85 at
    # least, 0.85-0.89 on 3 of the 33). With the client's search and stab (before the perception track), on the
    # same 33: placed 0.79-1.0 (0.79 on nav_fight_basin's seed 8), placed right 0.65-0.87 (0.73, 0.78 and 0.74
    # on the pinned seeds); counting every monster a throw can have hit places the same kills as the nearest
    # first did, on all 33. With the perception track too, its monsters walking as fast as the game
    # data says (a pig 125 units/s, not 70) and the throws before the look before placing what the last ones
    # don't, over the same 33 runs: placed 0.96-1.0, placed right 0.50-0.76 (pooled 0.66-0.68; 0.71, 0.67 and
    # 0.58 on the pinned seeds, hunt_clear_basin's, tour_hunt_basin's and nav_fight_basin's). A throw's path
    # is taken from the look before it, and faster monsters walk further from where that look saw them by the
    # time its star lands. So placed is back at 0.9, and placed right is 0.55 (two of the 33 below it: 0.50
    # and 0.55). With the client's search and stab on top of all three tracks, the same 33: placed 0.98-1.0,
    # placed right 0.46-0.83 (pooled 0.66-0.72; 0.78, 0.76 and 0.60 on the pinned seeds), one below the bar:
    # nav_fight_basin's seed 4, 0.46. With the game's 30 ms steps (motion.body.STEP) and jumps from standing
    # 150 ms after the arrow, the same 33: placed 0.93-1.0, placed right 0.40-0.82 (pooled 0.68; 0.61, 0.60 and
    # 0.54 on the pinned seeds): nav_fight_basin's seed 2 a kill under the 0.55, the pooled share as before. So
    # placed right is 0.5 (one of the 33 below it: nav_fight_basin's seed 7, 0.40). On the play track, before
    # the steps, with her finishing what she has hurt before she moves on (fight.owed), the same 33: placed
    # 0.97-1.0, placed right 0.42-0.83, pooled 0.70 as before (0.69), but three below 0.55: nav_fight_basin's
    # pinned seed, and its seeds 2 and 8 (0.53, 0.53, 0.42). So 0.5 there too. With her EXP number read, each gain
    # naming its kind and a kill put only on a monster of that kind (ledger.KIND_ANY off), over seeds 1-10 of each:
    # placed 0.79-1.0 (pooled 0.90), placed right 0.91-1.0 (pooled 0.96), and of all the kills 0.86 right against
    # 0.70 (the pinned seeds: placed 0.91, 0.90 and 0.88, right 1.0, 0.98 and 0.93). The bars: 0.75 and 0.9.
    assert len(placed) >= 0.75 * len(kills) and right >= 0.9 * len(placed)


# --- her wounds ---------------------------------------------------------------------------------------------

def _seen_at(t, monsters, x=500.0):
    """A look that searched for monsters, her at map x on a long flat platform (y 500)."""
    from gamelab.localization.mapdata import Platform, Segment
    from gamelab.localization.pose import Where
    p = Platform(0, [Segment(0, 500, 1000, 500)])
    s = _look(t)
    s.where, s.monsters, s.searched = Where(x, 500.0, p, (0.0, 0.0)), list(monsters), True
    return s


def _mon(kind, x, bar=None):
    from gamelab.localization.monsters import Monster
    from gamelab.perception.mobs import Mob
    return Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, 500.0, 0, bar)


def test_a_monster_in_her_throws_path_whose_bar_shows_is_hers_until_a_kill_of_hers_is_placed_on_it():
    from gamelab.behavior.ledger import Wounds
    h = _Hunter()
    ledger = Ledger(h)
    wounds = Wounds(ledger)
    pig, other = _mon("pig", 700.0), _mon("slime", 200.0, bar=0.5)  # the slime's bar: someone else's doing
    wounds.on_look(_seen_at(10.0, [pig, other]))
    ledger.on_look(_seen_at(10.0, [pig, other]))
    for w in (ledger, wounds):
        w.on_throw(10.1, "right")
    for w in (ledger, wounds):
        w.on_look(_seen_at(10.8, [_mon("pig", 690.0, bar=0.67), other]))
    assert [(m.kind, m.x) for m in wounds.seen_within(10.8, 1.8)] == [("pig", 690.0)]  # not the slime
    # Followed look to look, as long as it's seen; unseen (hidden under its damage numbers) it's still hers.
    for t, seen in ((11.2, [_mon("pig", 660.0, bar=0.67)]), (11.6, [])):
        for w in (ledger, wounds):
            w.on_look(_seen_at(t, seen))
    assert [m.x for m in wounds.seen_within(11.6, 1.8)] == [660.0]
    # A kill of hers placed on it closes it.
    h.exp_gains = 1
    for w in (ledger, wounds):
        w.on_look(_seen_at(12.0, []))
    assert ledger.kills[0].kind == "pig" and wounds.open == []


def test_a_wound_is_given_up_out_of_sight_and_after_being_chased_too_long():
    from gamelab.behavior.ledger import WOUND_LOST_S, WOUND_S, Wounds
    ledger = Ledger(_Hunter())
    wounds = Wounds(ledger)
    wounds.on_look(_seen_at(0.0, [_mon("pig", 700.0)]))
    wounds.on_throw(0.1, "right")
    wounds.on_look(_seen_at(0.8, [_mon("pig", 700.0, bar=0.6)]))
    wounds.on_look(_seen_at(0.8 + WOUND_LOST_S + 0.1, []))
    assert wounds.open == []  # out of sight that long: dead, or gone where she can't follow
    wounds.on_look(_seen_at(20.0, [_mon("pig", 700.0)]))
    wounds.on_throw(20.1, "right")
    for t in (20.8, 21.6, 22.4):
        wounds.on_look(_seen_at(t, [_mon("pig", 700.0, bar=0.6)]))
    assert len(wounds.open) == 1
    wounds.on_look(_seen_at(20.8 + WOUND_S + 0.1, [_mon("pig", 700.0, bar=0.5)]))
    assert wounds.open == []  # chased too long
    # A bar that shows on a monster no throw of hers had in its path isn't hers.
    wounds.on_look(_seen_at(60.0, [_mon("pig", 700.0, bar=0.6)]))
    assert wounds.open == []


def test_a_kill_is_placed_where_its_monster_was_seen_last_and_its_pile_expected_there():
    # A slime shot from 16 walked 90 units on 17 before it died: placed where it was at the throw, its pile lay
    # past LOOT_REACH of the drops, which were then taken for someone else's.
    h, store = _Hunter(), DropStore()
    ledger = Ledger(h, store)
    ledger.on_look(_seen_at(10.0, [_mon("slime", 800.0)]))
    ledger.on_throw(10.1, "right")
    for t, x in ((10.5, 840.0), (10.9, 880.0)):
        ledger.on_look(_seen_at(t, [_mon("slime", x)]))
    h.exp_gains = 1
    ledger.on_look(_seen_at(11.3, []))
    assert [(k.kind, k.x) for k in ledger.kills] == [("slime", 880.0)] and store.piles[0].x == 880.0


def test_a_pile_seen_holds_what_was_seen_in_it_and_what_that_is_worth():
    from gamelab.behavior.ledger import PILE_WORTH
    store = DropStore()
    store.on_kill(Kill(0.0, "slime", 3, 500.0, 300.0, 3))
    (p,) = store.known
    assert p.worth is None and p.value == PILE_WORTH  # not seen yet: a pile's coins
    seen = [Loot(Item("mesos", 0, 0, 13, 13, 0.9, "bronze"), 510.0, 300.0, 3),
            Loot(Item("slime-bubble", 0, 0, 19, 18, 0.9), 520.0, 300.0, 3)]
    store.on_look(_look(SETTLE, loot=seen, looked=[(3, 300.0, 700.0)]))
    assert p.items == ("mesos", "slime-bubble") and p.worth == p.value == 65 and p.dearest == 50 and p.mesos
    store.on_look(_look(SETTLE + 1, loot=seen[:1], looked=[(3, 300.0, 700.0)]))  # the bubble picked up
    assert p.items == ("mesos",) and p.worth == 15 and p.dearest == 15
    store.on_look(_look(SETTLE + 2, loot=[Loot(Item("leather", 0, 0, 20, 18, 0.9), 900.0, 300.0, 3)],
                        looked=[(3, 300.0, 1000.0)]))
    other = store.known[-1]
    assert not other.hers and other.items == ("leather",) and other.worth == 10  # someone else's, seen


def test_a_gain_goes_on_a_monster_gone_by_another_player_only_when_none_else_has_gone():
    """PLAYER_KILLS: of the monsters a throw of hers can have taken, gone from view, one another player stands by
    (they likelier killed it) comes after the others."""
    from gamelab.behavior.ledger import _Throw
    from gamelab.localization.monsters import Monster
    from gamelab.localization.players import Player
    from gamelab.perception.mobs import Mob

    def seen(kind, x):
        return Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, 500.0, 3)

    for players, kind in (([], "pig"), ([Player(1, "stranger", 250.0, 500.0, 3, (0.0, 0.0), 0.0, 1.0, seen=2)], "slime"),
                          ([Player(1, "party", 250.0, 500.0, 3, (0.0, 0.0), 0.0, 1.0, seen=2)], "pig")):
        h = _Hunter()
        ledger = Ledger(h)
        ledger._throws.append(_Throw(10.0, "right", [seen("pig", 300.0), seen("slime", 800.0)]))
        h.exp_gains = 1
        s = _look(10.5)
        s.searched, s.players = True, players  # both gone from view; the pig 50 units from the other player
        ledger.on_look(s)
        assert [k.kind for k in ledger.kills] == [kind], players
