"""The hunt's bookkeeping (behavior.ledger): every EXP gain a kill, placed where a star of hers explains it,
and every pile she knows of, kept from every look and every throw, whoever makes them. Nothing decides
from them yet (step 4 of docs/hunt-modes.md), so they're checked against the simulated world."""

import math

import numpy as np
import pytest

from gamelab.behavior.ledger import DROP_LANDS, GONE_KEEP_S, DropStore, Kill, Ledger
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
    sim.clock.sleep(DROP_LANDS)  # (a throw after the kill came in is dropped before its key: no lock waited out)
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
    from gamelab.behavior.looting import FIRST, expected_worth
    store = DropStore()
    store.on_kill(Kill(0.0, "slime", 3, 500.0, 300.0, 3))
    (p,) = store.known
    assert p.worth is None and p.value == expected_worth("slime", FIRST)  # not seen yet: its kind's expected drops
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


def test_a_gain_soon_after_a_throw_is_its_first_targets_seen_or_not():
    """A kill's EXP gain shows 0.15-0.46 s after the key of the throw that made it, before her star leaves
    (localization.monsters.GAIN_AFTER_KEY): the monster is still drawn, dying. So it's that throw's first of the kind in
    its path, though another of the kind in a recent throw's path is gone from view (until 2026-10-26 the gone one came
    first: 10 of 72 piles live went to the wrong monster). Without the gain's time, or later than that, as before."""
    from gamelab.behavior.ledger import DIE_S, DYING_DX, _Throw, dying
    DYING_S = DIE_S  # (a kind with no death animation in the game data)

    class Told(_Hunter):
        at: float | None = None

        def gains_told(self, n):
            return [(self.at, k) for k in self.kinds_told(n)]

    for at, x in ((10.3, 300.0), (10.9, 450.0), (None, 450.0)):
        h = Told()
        h.at, h.named = at, ["pig"]
        ledger = Ledger(h)
        ledger._throws += [_Throw(9.4, "right", [_mon("pig", 450.0)]), _Throw(10.0, "right", [_mon("pig", 300.0)])]
        h.exp_gains = 1
        ledger.on_look(_seen_at(10.95, [_mon("pig", 302.0)]))  # the first throw's pig is gone, the second's still drawn
        assert [(k.kind, k.x) for k in ledger.kills] == [("pig", x)], at
    k = Kill(10.0, "pig", 0, 300.0, 500.0, 0)
    assert dying([k], _mon("pig", 300.0 + DYING_DX - 1), 10.0 + DYING_S - 0.01)
    assert not dying([k], _mon("pig", 300.0 + DYING_DX + 1), 10.5) and not dying([k], _mon("slime", 300.0), 10.5)
    assert not dying([k], _mon("pig", 300.0), 10.0 + DYING_S + 0.01)
    assert dying([k], _mon("pig", 300.0), 10.0 + 1.2, die={"pig": 1.26})  # (a snail's 1.26, say: longer)
    # Of two of its kind by the kill, only the nearer is the one dying: the other is a live one beside it.
    near, beside = _mon("pig", 305.0), _mon("pig", 330.0)
    assert dying([k], near, 10.5, [near, beside]) and not dying([k], beside, 10.5, [near, beside])


def test_on_another_map_a_pile_is_worth_its_drops_there():
    # Tree Dungeon's monsters drop 25-53 mesos: a bronze coin there stands for 33.4 (looting.coins_for); a pile of a
    # Horny Mushroom's not seen yet its expected drops there (looting.expected_worth); its items are worth their shop
    # prices, as on every map.
    from gamelab.behavior.looting import Worth, expected_worth
    store = DropStore(Worth({"bronze": 33.4, "gold": 51.5}))
    store.on_kill(Kill(0.0, "horny-mushroom", 3, 500.0, 300.0, 3))
    (p,) = store.known
    assert p.worth is None and p.value == expected_worth("horny-mushroom", store.value) == pytest.approx(23.5, abs=0.1)
    seen = [Loot(Item("mesos", 0, 0, 13, 13, 0.9, "gold"), 510.0, 300.0, 3),
            Loot(Item("horny-mushroom-cap", 0, 0, 30, 27, 0.9), 520.0, 300.0, 3)]
    store.on_look(_look(SETTLE, loot=seen, looked=[(3, 300.0, 700.0)]))
    assert p.worth == p.value == 51.5 + 22 and p.dearest == 51.5
    store.on_look(_look(SETTLE + 1, loot=[Loot(Item("mesos", 0, 0, 13, 13, 0.9), 900.0, 300.0, 3)],
                        looked=[(3, 300.0, 1000.0)]))
    assert store.known[-1].worth == 33.4 and store.known[-1].unseen == 33.4  # someone else's, seen


def test_with_the_feed_a_kill_is_on_the_track_its_gain_is_bound_to_followed_and_its_pile_with_it():
    """With the feed's tracks (Scene.tracks), a gain is put on the track the tracker bound it to (Track.gain:
    localization.tracking, the monster her attack took at the key), not a live one beside it; its place and its pile
    follow that track as it walks on (it's drawn alive until her star lands); that track is dead for every throw
    (`dying`, Ledger.dead) by its id, the one beside it isn't, however near; and a gain the tracker puts on another (the
    first let go: seen alive since) goes with it."""
    from dataclasses import replace

    from gamelab.behavior.ledger import dying
    from gamelab.localization.monsters import Monster
    from gamelab.localization.tracking import DYING, LIVE, Track
    from gamelab.perception.mobs import Mob

    mob = Mob("pig", 0, 0, 30, 30, 0.9)

    def track(tid, x, state=LIVE, gain=None):
        return Track(tid, "pig", 3, x, 500.0, mob, 0.0, 10.0, state=state, gain=gain)

    class H(_Hunter):
        def gains_told(self, n):
            return [(10.2, "pig")] * n

    h, store = H(), DropStore()
    ledger = Ledger(h, store)
    h.exp_gains = 1
    s = _look(10.4)
    s.tracks = [track(5, 400.0, DYING, 10.2), track(6, 420.0)]
    s.monsters, s.searched = [Monster(mob, 400.0, 500.0, 3, track=5), Monster(mob, 420.0, 500.0, 3, track=6)], True
    ledger.on_look(s)
    (k,) = ledger.kills
    assert (k.kind, k.x, k.track) == ("pig", 400.0, 5) and [p.x for p in store.piles] == [400.0]
    assert set(ledger.dead) == {5}
    near, dead = s.monsters[1], s.monsters[0]
    assert dying(ledger.kills, dead, 10.5, s.monsters, None, ledger.dead)
    assert not dying(ledger.kills, near, 10.5, s.monsters, None, ledger.dead)  # 20 units off, alive: not by space
    s2 = _look(10.8)
    s2.tracks = [track(5, 450.0, DYING, 10.2), track(6, 420.0)]
    ledger.on_look(s2)
    assert (k.x, store.piles[0].x) == (450.0, 450.0)  # walked on: followed, its pile with it
    s3 = _look(11.6)
    s3.tracks = [track(5, 470.0), track(6, 420.0, DYING, 10.2)]  # the tracker let 5 go and put the gain on 6
    ledger.on_look(s3)
    assert (k.track, k.x) == (6, 420.0) and set(ledger.dead) == {6}
    s4 = _look(11.9)
    s4.tracks = [replace(t) for t in s3.tracks]
    s4.tracks[1].gain = None  # let go too, the gain on nobody: the kill on no track (0), not by space either
    ledger.on_look(s4)
    assert k.track == 0 and not ledger.dead
    assert not dying(ledger.kills, Monster(mob, 425.0, 500.0, 3, track=9), 12.0, [], None, ledger.dead)


def test_with_the_feed_her_kills_unsure_until_seen_dead_and_another_players_kills_theirs():
    """Ledger.dead's tracks told apart: one bound to her gain with no death seen or judged yet is `unsure` (it may be a
    live one: fight.TOUCH_NEAR keeps it in her body's way), its death seen or judged makes it sure; one dead by another's
    hand (the tracker's cause "other", or a share) is `theirs` (her memory's 'taken'), and no longer once it's hers
    after all (tracking._taken_back)."""
    from dataclasses import replace

    from gamelab.localization.tracking import DEAD, DYING, Track
    from gamelab.perception.mobs import Mob

    mob = Mob("pig", 0, 0, 30, 30, 0.9)

    def track(tid, x, state, gain=None, cause=None, seen=None):
        return Track(tid, "pig", 3, x, 500.0, mob, 0.0, 10.0, state=state, gain=gain, cause=cause, died_seen=seen)

    ledger = Ledger(_Hunter(), DropStore())
    s = _look(10.4)
    s.tracks = [track(5, 400.0, DYING, 10.2), track(7, 600.0, DEAD, cause="other", seen=10.3)]
    ledger.on_look(s)
    assert set(ledger.dead) == {5, 7} and ledger.unsure == {5} and ledger.theirs == {7}
    s2 = _look(10.9)
    s2.tracks = [replace(s.tracks[0], died_seen=10.8), replace(s.tracks[1], cause="hers", gain=10.2)]
    ledger.on_look(s2)
    assert set(ledger.dead) == {5, 7} and not ledger.unsure and not ledger.theirs


# --- the drop store for the loot sweep (2026-10-30) -------------------------------------------------------------------

def _coin(x, platform=19):
    return Loot(Item("mesos", 0, 0, 13, 13, 0.9, "bronze"), x, -200.0, platform)


def test_a_drop_seen_is_in_one_pile_only_the_nearest():
    """Live on 2026-10-30 the thief's run listed three piles at x 254, 256 and 260 on platform 19 (two kills of hers
    there and one seen first), each taking the drops seen within LOOT_REACH of it: 15-24% of the piles' worth listed was
    the same drops counted by 2-3 piles. Each drop now goes to the nearest pile alone."""
    store = DropStore()
    store.on_kill(Kill(0.0, "green-mushroom", 19, 254.0, -200.0, 19))
    store.on_kill(Kill(0.5, "green-mushroom", 19, 260.0, -200.0, 19))
    a, b = store.piles
    store.on_look(_look(SETTLE + 1, loot=[_coin(250.0), _coin(258.0), _coin(262.0)], looked=[(19, 100.0, 400.0)]))
    assert a.xs == (250.0,) and b.xs == (258.0, 262.0) and len(store.piles) == 2
    assert a.value + b.value == pytest.approx(3 * store.value.coins["bronze"])  # (each pile had all three: 6 coins)


def test_drops_no_pile_has_make_piles_of_their_own_a_loot_reach_apart():
    store = DropStore()
    store.on_look(_look(5.0, loot=[_coin(100.0), _coin(120.0), _coin(150.0), _coin(300.0)],
                        looked=[(19, 0.0, 400.0)]))
    assert [(round(p.x), p.xs) for p in store.piles] == [(125, (100.0, 120.0, 150.0)), (300, (300.0,))]
    assert all(not p.hers and p.status == "seen" for p in store.piles)


def test_the_states_strips_of_drops_count_each_at_its_own_time():
    """Every strip of drops perception searched since the look before (the state's rotation: Scene.strips) is a
    sighting at its own time: a pile seen in one is confirmed; one in a strip searched empty twice EMPTY_LOOKS apart is
    gone; a strip searched before the look that last saw a pile says nothing of it."""
    from gamelab.localization.worldstate import DropStrip
    store = DropStore()
    store.on_kill(Kill(0.0, "slime", 3, 500.0, 300.0, 3))
    (p,) = store.piles
    s = _look(SETTLE + 1.0)
    s.strips = [DropStrip(3, 400.0, 600.0, SETTLE + 0.5, False, (Loot(Item("mesos", 0, 0, 13, 13, 0.9), 505.0, 300.0,
                                                                      3),))]
    store.on_look(s)
    assert p.status == "seen" and p.last_seen == SETTLE + 0.5 and p.xs == (505.0,)
    s = _look(SETTLE + 5.0)  # the same strip again (no newer search): nothing new
    s.strips = list(s.strips) + [DropStrip(3, 400.0, 600.0, SETTLE + 0.5)]
    store.on_look(s)
    assert p.empty_since is None
    for t in (SETTLE + 6.0, SETTLE + 6.0 + EMPTY_LOOKS):
        s = _look(t + 0.5)
        s.strips = [DropStrip(3, 400.0, 600.0, t)]
        store.on_look(s)
    assert p.status == "gone" and p.gone_at == SETTLE + 6.0 + EMPTY_LOOKS


def test_only_a_strip_sure_of_it_sees_a_pile_empty_and_a_missed_pile_is_hers_again():
    """The review of 2026-10-30: the rotation's strips, searched mid-hop (placed by her offset in the air, as high above
    their ground as she is) or cut short by the bottom bar, had 15 of 49 of her piles seen gone with their coins still
    there (seed 3; master 1 of 41). A strip that isn't sure of it (worldstate.DropStrip.sure) sees no pile empty, however
    often; with the state's strips the look's own sighting confirms what it saw and no more (its strips are among them);
    drops seen again within LOOT_REACH of a pile of hers seen gone REVIVE_S ago at most are that pile's again (hers,
    worth what's seen), not another's; later, another's."""
    from gamelab.behavior.ledger import REVIVE_S
    from gamelab.localization.worldstate import DropStrip
    store = DropStore()
    store.on_kill(Kill(0.0, "slime", 3, 500.0, 300.0, 3))
    (p,) = store.piles
    for t in (SETTLE + 1.0, SETTLE + 1.0 + EMPTY_LOOKS, SETTLE + 1.0 + 2 * EMPTY_LOOKS):
        s = _look(t + 0.1, looked=[(3, 400.0, 600.0)])  # (the look's own search: its strip unsure among the state's)
        s.strips = [DropStrip(3, 400.0, 600.0, t + 0.1, sure=False), DropStrip(3, 600.0, 700.0, t)]
        store.on_look(s)
    assert p.status == "expected" and p.empty_since is None
    for t in (SETTLE + 6.0, SETTLE + 6.0 + EMPTY_LOOKS):  # sure ones see it empty as before
        s = _look(t + 0.5)
        s.strips = [DropStrip(3, 400.0, 600.0, t)]
        store.on_look(s)
    assert p.status == "gone"
    gone = p.gone_at
    s = _look(gone + 2.5)
    s.strips = [DropStrip(3, 400.0, 600.0, gone + 2.4, False, (Loot(Item("mesos", 0, 0, 13, 13, 0.9), 520.0, 300.0, 3),))]
    store.on_look(s)
    assert store.piles == [p] and p.hers and p.status == "seen" and p.xs == (520.0,) and p.gone_at is None
    for t in (gone + 4.0, gone + 4.0 + EMPTY_LOOKS):  # gone again ...
        s = _look(t + 0.5)
        s.strips = [DropStrip(3, 400.0, 600.0, t)]
        store.on_look(s)
    later = p.gone_at + REVIVE_S + 1.0  # ... and drops there past REVIVE_S: another's
    s = _look(later + 0.1)
    s.strips = [DropStrip(3, 400.0, 600.0, later, False, (Loot(Item("mesos", 0, 0, 13, 13, 0.9), 510.0, 300.0, 3),))]
    store.on_look(s)
    assert p.status == "gone" and [q.hers for q in store.piles if q is not p] == [False]


def test_a_pile_lies_by_the_live_survival_not_a_fixed_life():
    """ledger.SURVIVAL by the live evidence (2026-10-31): hers lie whole for OWN_S (her ownership), then are taken by
    others at TAKE_RATE a minute (0.89 still there at 150 s), and go at the server's expiry pass past EXPIRE_S, any time
    in EXPIRE_PASS_S (the review of 2026-10-31: one seen lying past EXPIRE_S still there, likelier the sooner it was
    seen; until then none past it), none after; a pile of hers seen at 100 s is
    still there 20 s on more than 95 times in 100, where the raw curve (0.26 / 0.42 at 120 and 100 s) gave it 60;
    given it was seen a moment ago, about all of it; one never seen, from its kill. Others' are the raw curve still."""
    from gamelab.behavior.ledger import EXPIRE_PASS_S, EXPIRE_S, OWN_S, TAKE_RATE, Pile, survival
    assert survival(0.0) == survival(OWN_S) == 1.0 and survival(150.0) == pytest.approx(0.89, abs=0.01)
    assert survival(150.0) == pytest.approx(math.exp(-TAKE_RATE / 60.0 * (150.0 - OWN_S)))
    last = EXPIRE_S + EXPIRE_PASS_S
    assert survival(EXPIRE_S - 1.0) > 0.85 and survival(EXPIRE_S + EXPIRE_PASS_S / 2) == pytest.approx(
        survival(EXPIRE_S) / 2) and survival(last) == 0.0 and survival(last + 1.0) == 0.0
    late = Pile(0.0, 0.0, 3, 0.0, True, status="seen", last_seen=EXPIRE_S + 20.0)  # (seen lying past EXPIRE_S)
    assert late.still(EXPIRE_S + 20.0) == pytest.approx(1.0) and late.still(EXPIRE_S + 30.0) == pytest.approx(0.75)
    assert survival(60.0, hers=False) == pytest.approx(0.57) and survival(150.0, hers=False) == pytest.approx(0.03)
    p = Pile(0.0, 0.0, 3, 0.0, True, status="seen", last_seen=100.0)
    assert p.still(120.0) == pytest.approx(survival(120.0) / survival(100.0)) and p.still(120.0) > 0.95
    p.last_seen = 99.9
    assert p.still(100.0) == pytest.approx(1.0, abs=0.01) and p.still(100.0, 30.0) < p.still(100.0, 5.0)
    assert Pile(0.0, 0.0, 3, 0.0, True).still(10.0) == pytest.approx(survival(10.0))


def test_a_pile_of_hers_holding_an_older_one_s_drops_keeps_that_one_s_age():
    """HELD_AGE: every drop seen goes to its nearest pile, so a newer kill's pile by her older drops takes them and the
    older pile, seen empty twice, turns gone: the one holding its drops takes its age (the drops'), not the newer
    kill's; one of another's or too far off doesn't. Only one holding a drop where one of the older's lay (HELD_DX,
    the review of 2026-10-31): one by an older pile she picked up herself keeps its own age."""
    from gamelab.behavior.ledger import Pile
    store = DropStore()
    old = Pile(100.0, -200.0, 19, 10.0, True, status="seen", last_seen=40.0, xs=(125.0,))
    new = Pile(130.0, -200.0, 19, 90.0, True)
    far = Pile(400.0, -200.0, 19, 95.0, True)
    theirs = Pile(70.0, -200.0, 19, 5.0, False, status="seen", last_seen=40.0, xs=(70.0,))
    store.piles = [old, new, far, theirs]
    seen = [_coin(125.0), _coin(400.0)]  # (her old drops by 125: nearer the new kill's pile, it takes them)
    store._sighting(100.0, seen, lambda pid, x: True, True)
    assert new.status == far.status == "seen" and old.empty_since == 100.0 and new.since == 90.0
    store._sighting(100.0 + EMPTY_LOOKS, seen, lambda pid, x: True, True)
    assert old.status == "gone" and new.since == 10.0 and far.since == 95.0
    assert theirs.status == "gone" and new.since == 10.0  # (another's turning gone gives nobody its age)
    store = DropStore()  # (she picked the older pile at 100 up herself: the newer kill's drops lie at 140 and 150)
    old = Pile(100.0, -200.0, 19, 10.0, True, status="seen", last_seen=40.0, xs=(100.0, 105.0))
    new = Pile(130.0, -200.0, 19, 90.0, True)
    store.piles = [old, new]
    for t in (100.0, 100.0 + EMPTY_LOOKS):
        store._sighting(t, [_coin(140.0), _coin(150.0)], lambda pid, x: True, True)
    assert old.status == "gone" and new.status == "seen" and new.since == 90.0


def test_an_expected_pile_is_worth_its_kinds_drops():
    """Before a look sees it a kill's pile is worth what its kind drops (looting.expected_worth: the live 0.64 drops a
    kill, 57% mesos at the middle of its amounts, the rest its items by their rates), not a bronze coin: on Tree
    Dungeon a Horny Mushroom's 23.5, a Green's 14.1; a kind none of that knows, a bronze coin."""
    from gamelab.behavior.looting import Worth, expected_worth
    tree_worth = Worth({"bronze": 33.4, "gold": 51.5})
    store = DropStore(value=tree_worth)
    horny = store.on_kill(Kill(0.0, "horny-mushroom", 3, 500.0, 300.0, 3))
    green = store.on_kill(Kill(0.0, "green-mushroom", 3, 700.0, 300.0, 3))
    pig = store.on_kill(Kill(0.0, "pig", 3, 900.0, 300.0, 3))
    assert horny.value == pytest.approx(23.5, abs=0.1) and green.value == pytest.approx(14.1, abs=0.1)
    assert pig.value == 33.4 and horny.kind == "horny-mushroom"
    assert expected_worth("horny-mushroom", tree_worth) == horny.value
