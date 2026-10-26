"""The monsters about her that a look didn't find (localization.unseen) and what's decided by them: the hunt loop's
fights (Hunt._with_unseen: around her, and those on her platform that count as a danger as targets) and her pickups
(Looter.unseen). On a synthetic map of Tree Dungeon's kinds (test_danger's): no game, no simulator."""

import time
from types import SimpleNamespace

from gamelab.attack.danger import Danger
from gamelab.behavior.fight import FightState
from gamelab.behavior.hunt import Hunt
from gamelab.behavior.looting import DropSpot, Looter
from gamelab.localization import unseen as unseen_mod
from gamelab.localization.monsters import Monster
from gamelab.localization.tracking import DYING, HIDDEN, LIVE, Track
from gamelab.localization.unseen import UNSEEN_S, unseen
from gamelab.perception.mobs import Mob

from test_danger import HER, build


def _map():
    """Her platform (0..1000, y 500) with a Horny Mushroom's spawn point at 800, and one 200 above (0..1000, y 300)."""
    return build([(0, 500, 1000, 500), (0, 300, 1000, 300)], [("19", 800, 500)])


def _mob(kind="horny-mushroom"):
    return Mob(kind, 0, 0, 30, 30, 0.9)


def _seen(kind, x, pid, model):
    return Monster(_mob(kind), x, model.platform(pid).y_at(x), pid)


def _track(i, x, pid, last_seen, vx=0.0, state=HIDDEN, kind="horny-mushroom", owed=False):
    return Track(i, kind, pid, x, 500.0, _mob(kind), last_seen - 5.0, last_seen, vx=vx, state=state, owed=owed)


def _entry(x, pid, seen, model, kind="horny-mushroom"):
    return SimpleNamespace(kind=kind, platform=pid, x=x, seen=seen, monster=_seen(kind, x, pid, model))


def _scene(t, x, pid, monsters=(), tracks=(), swept=()):
    return SimpleNamespace(t=t, where=SimpleNamespace(x=x, y=500.0, platform=SimpleNamespace(id=pid)),
                           platform=SimpleNamespace(id=pid), monsters=list(monsters), swept=list(swept),
                           tracks=list(tracks))


def test_unseen_are_the_feeds_hidden_tracks_walked_on_and_her_memorys_last_moments():
    m = _map()
    her, up = 1, 0  # (platform ids: top down)
    assert m.platform(her).y_at(500) == 500
    t = 100.0
    tracks = [_track(1, 300.0, her, t - 1.0, vx=-50.0),  # hidden a second: walked on 50 units left
              _track(2, 600.0, her, t - unseen_mod.TRACK_S - 0.5),  # out of sight too long: not counted
              _track(3, 700.0, her, t - 0.2, state=LIVE), _track(4, 650.0, her, t - 0.2, state=DYING)]
    memory = SimpleNamespace(alive=lambda at, p: [_entry(900.0, her, t - 1.5, m), _entry(950.0, her, t - UNSEEN_S - 0.5, m),
                                                  _entry(310.0, her, t - 0.5, m), _entry(200.0, up, t - 1.0, m)])
    s = _scene(t, 500.0, her, monsters=[_seen("horny-mushroom", 880.0, up, m)], tracks=tracks)
    got = unseen(s, memory, m)
    assert [(g.platform, round(g.x)) for g in got] == [(her, 250), (her, 900), (up, 200)]
    # the entry at 310 is the hidden track's (it can have walked there); one the look found is that one
    s.monsters = [_seen("horny-mushroom", 920.0, her, m)]
    assert [(g.platform, round(g.x)) for g in unseen(s, memory, m)] == [(her, 250), (up, 200)]
    assert [round(g.x) for g in unseen(s, memory, m, platforms={up})] == [200]
    assert unseen(s, None, m) == got[:1]
    assert got[1].pose == ""  # (its walking not known)


def test_a_track_her_kill_is_on_isnt_about_one_kill_one_track():
    """unseen.hidden: of the feed's tracks out of sight, not the one a kill of hers in the ledger is on (of its kind on
    its platform, the nearest where it was placed within KILL_NEAR, seen about when it was killed; one not seen live
    first), nor one the feed says her waiting gain goes on (Track.owed), nor one out of sight past TRACK_S; a second
    one beside it stays (one kill, one track), and so does one seen there well after the kill (another one)."""
    m = _map()
    her = 1
    t = 100.0
    kill = SimpleNamespace(t=t - 0.6, kind="horny-mushroom", platform=her, x=400.0)
    tracks = [_track(1, 430.0, her, t - 0.3), _track(2, 470.0, her, t - 0.4),  # the kill's, and one beside it
              _track(3, 410.0, her, t - 1.2, kind="green-mushroom"),  # another kind
              _track(4, 300.0, her, t - 0.5, owed=True)]  # the feed's: her gain waits on it
    s = _scene(t, 500.0, her, tracks=tracks)
    assert [tr.id for tr in unseen_mod.hidden(s, [kill])] == [2, 3]
    assert [tr.id for tr in unseen_mod.hidden(s)] == [1, 2, 3]
    live = _track(5, 405.0, her, t - 0.05, state=LIVE)  # seen now beside where it died: not the one dead
    s.tracks = tracks + [live]
    assert [tr.id for tr in unseen_mod.hidden(s, [kill])] == [2, 3]
    earlier = SimpleNamespace(t=t - 3.0, kind="horny-mushroom", platform=her, x=470.0)
    s.tracks = [_track(6, 470.0, her, t - 1.0)]  # seen 2 s after that kill there: another one, walking in
    assert [tr.id for tr in unseen_mod.hidden(s, [earlier])] == [6]
    gone = SimpleNamespace(t=t - 9.0, kind="horny-mushroom", platform=her, x=430.0)  # long ago: no matter
    s.tracks = tracks
    assert [tr.id for tr in unseen_mod.hidden(s, [gone, kill])] == [2, 3]
    assert [round(g.x) for g in unseen(s, None, m, kills=[kill])] == [470, 410]


def _hunt(model, unseen_list, targets=()):
    h = object.__new__(Hunt)
    h.h = SimpleNamespace(danger=Danger(model, HER), memory=None)
    h.model, h.fs = model, FightState()
    h._unseen_for, h._unseen_list = None, []
    h._unseen = lambda s: list(unseen_list)
    return h


def test_her_fights_go_by_the_monsters_about_her_the_look_didnt_find():
    """Hunt._with_unseen: all of them around her (the stance's `around`, as hidden ones), and those on her platform of a
    kind that counts as a danger among her targets too (rule 1 keeps off them, the stance shoots them first); not
    those on other platforms (thrown at where last seen, they're the throws at what's gone), nor one she threw at
    already (the fight's own hidden ones)."""
    m = _map()
    her, up = 1, 0
    close = _seen("horny-mushroom", 530.0, her, m)
    above = _seen("horny-mushroom", 500.0, up, m)
    thrown = _seen("horny-mushroom", 300.0, her, m)
    h = _hunt(m, [close, above, _seen("horny-mushroom", 310.0, her, m)])
    s = _scene(10.0, 500.0, her)
    targets, visible, hidden = h._with_unseen(s, [thrown], [], [thrown])
    assert targets == [thrown, close] and hidden == [thrown, close, above] and visible == []
    h.h.danger = None  # where nothing counts (the first map): as they were
    assert h._with_unseen(s, [thrown], [], [thrown]) == ([thrown], [], [thrown])


def test_a_drop_guarded_by_a_monster_the_look_missed_waits_too():
    """Looter.unseen (the hunt loop's Hunt._unseen): a monster about that the look didn't find guards drops as the look's
    own do (looting's TOO_CLOSE)."""
    m = _map()
    her = 1
    said = []
    s = SimpleNamespace(where=SimpleNamespace(x=500.0), platform=SimpleNamespace(id=her), monsters=[], loot=[],
                        in_view=lambda pid, x: True)
    h = SimpleNamespace(look=lambda loot=False: s, log=said.append, report=SimpleNamespace(loot_runs=0))
    looter = Looter(h)
    looter.sweep_through = lambda *a, **k: said.append("swept")
    spot = DropSpot(560.0, 500.0, her, time.monotonic() - 1.0, True, on_ground=False)
    assert looter.collect(spot) and said == ["swept"]
    looter.unseen = lambda look: [_seen("horny-mushroom", 600.0, her, m)]
    said.clear()
    assert not looter.collect(spot) and said == ["  drops guarded by a monster: later"] and looter.pending == [spot]
