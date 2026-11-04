"""Her throws at monsters no look has go by the evidence of each look and their price (behavior.presence, the thief's: Attacks
.worth_a_throw, fight.shots_at), and the frozen scenes of the user's report of 2026-11-04 ("mobs near the player are not
detected by perception and kept for 2s, during which the motion planner will keep attempting attack the mob even tho it
may well have died"): no game, no simulator."""

import time
from types import SimpleNamespace

import pytest
from maplehelper.engine import LEFT, RIGHT
from maplehelper.keys import parse as K

from gamelab.attack.danger import Danger
from gamelab.attack.stance import stance
from gamelab.behavior import attacks as attacks_mod
from gamelab.behavior import fight, hunt, presence
from gamelab.behavior.attacks import LANDS_S, SHOWS_S, Attacks
from gamelab.behavior.fight import FightState
from gamelab.behavior.hunt import _same
from gamelab.behavior.presence import Evidence, chance
from gamelab.localization.monsters import Monster
from gamelab.localization.tracking import DYING, HIDDEN, LIVE, Track
from gamelab.perception.mobs import Mob

from test_danger import HER, build

MODEL = SimpleNamespace(mob_hp={"Green Mushroom": 233.0, "Horny Mushroom": 403.0},
                        mob_pdd={"Green Mushroom": 0.0, "Horny Mushroom": 10.0},
                        mob_exp={"Green Mushroom": 26, "Horny Mushroom": 41}, spawns=[], id=10002073)
KEYS = SimpleNamespace(attack=K("d"), lucky_seven=K("a"))
MAX_MP = 800.0
RATE = 6.85


def _attacks(mode="choice", **kw):
    said = []
    a = Attacks(MODEL, KEYS, mode, max_mp=MAX_MP, level=20, log=said.append, **kw)
    a.rate = lambda: RATE
    a.from_ground = lambda: True
    return a, said


def _monster(kind, x, bar=None, track=None):
    return Monster(Mob(kind, 0, 0, 0, 0, 1.0), x, 0.0, 1, bar, track=track)


def _track(i, x, last_seen, state=HIDDEN, kind="horny-mushroom", seen=40, blind=0, bar=None):
    return Track(i, kind, 1, x, 0.0, Mob(kind, 0, 0, 0, 0, 1.0), last_seen - 5.0, last_seen, seen=seen, state=state,
                 blind=blind, bar=bar)


def _look(monsters, t, tracks=(), mp=1.0):
    """A look of her standing at x 0 on platform 1 (y 0), with `monsters` and the feed's `tracks`."""
    p = SimpleNamespace(id=1, walls=(), floors=(), peers=(), left=-500.0, right=500.0, y_at=lambda x: 0.0)
    return SimpleNamespace(t=t, mp=mp, searched=True, where=SimpleNamespace(x=0.0, y=0.0, platform=p), platform=p,
                           monsters=list(monsters), tracks=list(tracks))


def _thrown_at_then_gone(a, kind="horny-mushroom", x=120.0, bar=None, ago=0.3, tracks=None, track=7):
    """A monster seen at x, thrown at while seen, `ago` s before `now`; the look at `now` no longer finds it: its feed
    track out of sight. Returns it as last seen."""
    now = time.monotonic()
    m = _monster(kind, x, bar, track)
    seen = now - ago
    a.on_look(_look([m], seen, [_track(track, x, seen, LIVE, kind)]))
    assert a.key(RIGHT) is not None
    a.on_throw(seen, RIGHT)
    a.marks[-1].seen = seen  # (the look it was thrown on, as a key that comes on the look's heels has it)
    tr = _track(track, x, seen, HIDDEN, kind) if tracks is None else tracks
    a.on_look(_look([], now, [tr] if tr else []))
    return m


# --- the chance -------------------------------------------------------------------------------------------------------


def test_each_piece_of_evidence_moves_the_chance_its_way():
    fresh = chance(Evidence(0.2, LIVE))
    assert 0.4 < fresh < 0.6  # (live, within 0.3 s of a sighting: 58% showed an effect)
    assert chance(Evidence(1.5, HIDDEN)) < fresh  # time
    assert chance(Evidence(0.2, "gone")) < fresh  # its track gone from the feed
    assert chance(Evidence(0.2, DYING)) < 0.2 and chance(Evidence(0.2, "dead")) < 0.2  # the feed has it dying
    assert chance(Evidence(0.2, LIVE, gain=True)) < fresh  # her EXP up: its kind's
    assert chance(Evidence(0.2, LIVE, blank=1)) < fresh < chance(Evidence(0.2, LIVE, proven=True))  # her last throw
    assert chance(Evidence(0.2, LIVE, short=True)) < fresh  # a flicker, or a dead walker's new id
    # her reckoned damage should have killed it, with a throw at it that showed nothing: her throws didn't land
    assert chance(Evidence(0.2, LIVE, blank=1, spent=0.9)) < chance(Evidence(0.2, LIVE, blank=1))
    assert chance(Evidence(0.2, LIVE, spent=0.9)) == fresh  # (... and with hits shown, it's a hit monster, no more)
    # the exact finder's misses: the first few say little (its gaps on the lossless clip), then fast
    assert chance(Evidence(0.2, LIVE, blind=2)) == fresh
    assert chance(Evidence(0.3, LIVE, blind=5)) < 0.05 < chance(Evidence(0.3, LIVE, blind=presence.BLIND_FREE))


def test_a_landing_is_worth_what_it_takes_a_throw_costs_its_mp_and_its_lock():
    assert presence.landing(50.0, 400.0, 200.0) == 25.0  # half of it
    assert presence.landing(50.0, 100.0, 200.0) == 50.0  # all of it
    assert presence.cost(16.0, 0.86, 7.0) == pytest.approx(16 * presence.MP_EXP + 0.86 * 7.0)
    assert presence.MESO_EXP == hunt.MESO_EXP  # (the user's prices, the hunt's)
    assert presence.net(0.5, 50.0, 100.0, 200.0, 0.0, 1.0, 5.0) == pytest.approx(20.0)


# --- the frozen scenes ------------------------------------------------------------------------------------------------


def test_one_truly_under_her_damage_numbers_is_thrown_at():
    """Just hit, hidden under its damage numbers a look or two: live 58% of those within 0.3 s of a sighting showed an
    effect, and that's what HIDDEN_FOR was for (2026-11-04 00:23:58.751: a real Horny Mushroom, killed so)."""
    a, _ = _attacks()
    m = _thrown_at_then_gone(a, bar=0.5, ago=0.3)
    assert a.chance(m) is not None and a.worth_a_throw(m)
    assert a.target(RIGHT) is not None and a.key(RIGHT) is not None


def test_one_with_exp_just_gained_isnt_worth_a_cast():
    """A gain of its kind since it went: most likely it (2026-11-04 00:25:02.214: fading, its gain 0.8 s before the key,
    bound to the live #59). Its gain of another kind says nothing."""
    a, _ = _attacks()
    m = _thrown_at_then_gone(a, ago=0.9)
    before = a.chance(m)
    a.heard(time.monotonic(), [(time.monotonic() - 0.2, frozenset({"green-mushroom"}))])
    assert a.chance(m) == pytest.approx(before, abs=0.05)
    a.heard(time.monotonic(), [(time.monotonic() - 0.1, frozenset({"horny-mushroom"}))])
    assert a.chance(m) < before - 0.15
    assert not a.worth_a_throw(m)
    assert a.target(RIGHT) is None


def test_a_throw_that_showed_nothing_counts_against_the_next():
    """The thief's rule 3: her throw at one no look had, with no effect by LANDS_S after its key, is one that showed
    nothing. 2026-11-04 00:24:00.537 and 01.384: a dead Horny Mushroom's ghost, two casts into nothing, the user's case."""
    a, _ = _attacks()
    m = _thrown_at_then_gone(a, bar=0.5, ago=0.2)
    t = time.monotonic()
    assert a.key(RIGHT) is not None  # (at it, unseen)
    a.on_throw(t, RIGHT)
    mk = a._mark_of(m)
    assert len(mk.blind) == 1 and mk.blank == 0
    fresh = a.chance(m)
    a.heard(t + LANDS_S + 0.05, [])
    a.on_look(_look([], t + LANDS_S + 0.05, [_track(7, 120.0, t - 0.2, HIDDEN)], mp=1.0 - 16 / MAX_MP))
    assert mk.blank == 1 and not mk.blind
    assert a.counts["unseen: showed nothing"] == 1
    assert a.chance(m) < fresh
    # Seen again, it owes nothing
    again = _monster("horny-mushroom", 110.0, 0.5, 7)
    a.on_look(_look([again], t + LANDS_S + 0.2, [_track(7, 110.0, t + LANDS_S + 0.2, LIVE)]))
    assert mk.blank == 0


@pytest.mark.parametrize("effect", ["gain", "bar", "claw", "miss"])
def test_a_throw_that_showed_something_lifts_the_next(effect):
    """An effect of it: any gain since (a kill, maybe another's), a bar lower or up that way, her claw (a ground cast that took
    no MP: a body there), her attack's MISS popup where it was (she picked it)."""
    a, _ = _attacks()
    m = _thrown_at_then_gone(a, bar=0.5, ago=0.2)
    t = time.monotonic()
    assert a.key(RIGHT) is not None
    a.on_throw(t, RIGHT)
    mk = a._mark_of(m)
    other = _monster("green-mushroom", 300.0, None, 9)
    tracks = [_track(7, 120.0, t - 0.2, HIDDEN)]
    gone_mp = 1.0 - 16 / MAX_MP
    a.on_look(_look([other], t + SHOWS_S / 2, tracks, mp=1.0))
    when = t + LANDS_S + 0.05
    if effect == "gain":
        a.heard(t + SHOWS_S + 0.1, [(t + SHOWS_S + 0.05, frozenset({"horny-mushroom"}))])
        a.on_look(_look([other], when, tracks, mp=gone_mp))
    elif effect == "bar":
        a.on_look(_look([_monster("green-mushroom", 300.0, 0.7, 9)], when, tracks, mp=gone_mp))
    elif effect == "claw":
        a.on_look(_look([other], when + attacks_mod.MP_SHOWS_S, tracks, mp=1.0))
    else:
        look = _look([other], when, tracks, mp=gone_mp)
        look.misses = [(120.0, -40.0)]
        a.on_look(look)
    assert mk.blank == 0 and mk.shown and not mk.blind
    assert a.chance(m) > presence.chance(Evidence(a.evidence(m).age, HIDDEN))


def test_one_whose_reckoned_hp_is_exhausted_is_let_go():
    """Reckoned dead by her throws, with one at it unseen that showed nothing and no EXP: they didn't land, and the next
    won't. (2026-11-04 00:24: 'HP left' went on falling, 214 then 182, at a ghost: the reckoning alone couldn't notice.)"""
    a, _ = _attacks()
    m = _thrown_at_then_gone(a, kind="green-mushroom", ago=0.3)  # (233 HP: a cast takes about all of it)
    t = time.monotonic()
    assert a.key(RIGHT) is not None
    a.on_throw(t, RIGHT)
    a.on_look(_look([], t + LANDS_S + 0.05, [_track(7, 120.0, t - 0.3, HIDDEN, "green-mushroom")], mp=1.0 - 16 / MAX_MP))
    e = a.evidence(m)
    assert e.blank == 1 and e.spent >= presence.SPENT_DEAD
    assert a.chance(m) < 0.25
    assert not a.worth_a_throw(m) and a.target(RIGHT) is None


def test_a_track_the_feed_lost_or_has_dying_isnt_thrown_at():
    """Its track gone from the feed (live 11% showed, 95% dead or gone) with a second gone, or dying by it."""
    a, _ = _attacks()
    m = _thrown_at_then_gone(a, ago=1.2, tracks=False)
    assert a.evidence(m).state == "gone" and a.evidence(m).age > 1.0 and not a.worth_a_throw(m)
    a, _ = _attacks()
    m = _thrown_at_then_gone(a, ago=0.3, tracks=_track(7, 120.0, time.monotonic() - 0.3, DYING))
    assert a.evidence(m).state == DYING and not a.worth_a_throw(m)


def test_the_finders_misses_soon_say_its_gone():
    """A monster the exact finder has missed on 5 frames in a row (0.33 s) isn't under anything it can see through."""
    a, _ = _attacks()
    m = _thrown_at_then_gone(a, ago=0.3, tracks=_track(7, 120.0, time.monotonic() - 0.3, HIDDEN, blind=5))
    assert a.evidence(m).blind == 5 and not a.worth_a_throw(m)


def test_a_cheap_regular_attack_goes_on_thinner_evidence_than_a_cast_and_a_nearly_dead_one_on_thinner_than_a_whole():
    a, _ = _attacks()
    m = _thrown_at_then_gone(a, bar=0.1, ago=1.3)  # (a Horny Mushroom at 40 HP)
    p = a.chance(m)
    assert p is not None and p < 0.5
    net, attack = a._best_net(m, p)
    assert net >= 0.0 and attack is a.regular
    assert a.key(RIGHT) == KEYS.attack.name
    whole, _ = _attacks()
    w = _thrown_at_then_gone(whole, ago=1.3)
    # (one whole: its landing is worth a share only, and at this chance no throw pays)
    assert whole._best_net(w, p)[0] < net


def test_a_throw_at_one_she_sees_comes_first():
    """Worth a throw on its own, no better than one at a monster in reach that a look has: a better target now."""
    a, _ = _attacks()
    m = _thrown_at_then_gone(a, ago=0.5)
    assert a.worth_a_throw(m)
    seen = _monster("horny-mushroom", -120.0, None, 8)
    assert not a.worth_a_throw(m, a.seen_net(seen))


def test_nothing_there_no_throw():
    """A throw that way with nothing a look has that way, none worth one unseen: no key (live 3-4% of her throws went so,
    the regular attack at nothing); with a monster in reach that way as before; with a fresh touch her claw may take what's
    on her."""
    a, said = _attacks()
    t = time.monotonic()
    a.on_look(_look([], t))
    assert a.key(RIGHT) is None and a.counts["no throw: nothing there"] == 1 and a.nothing_to_throw_at(RIGHT)
    a.touched = lambda s: True
    assert a.key(RIGHT) == KEYS.attack.name and not a.nothing_to_throw_at(RIGHT)
    a.touched = None
    a.on_look(_look([_monster("horny-mushroom", -120.0)], t + 0.1))
    assert a.key(RIGHT) is None  # (none that way)
    assert a.key(LEFT) == KEYS.lucky_seven.name
    assert fight.cast_or_wait(SimpleNamespace(attacks=a), ("throw", RIGHT)) == ("wait",)
    assert fight.cast_or_wait(SimpleNamespace(attacks=a), ("throw", LEFT)) == ("throw", LEFT)


def test_the_newest_state_before_the_key_counts():
    """Hunter._still tells her the feed's newest tracks: seen in that frame, no look's miss holds it; lost from it, it's gone."""
    a, _ = _attacks()
    m = _thrown_at_then_gone(a, ago=0.2)
    t = time.monotonic()
    a.newest(t, [_track(7, 125.0, t, LIVE)])
    assert a.chance(m) is None
    a.newest(t + 0.05, [_track(9, 400.0, t + 0.05, LIVE)])
    assert a.evidence(m).state == "gone"


def test_her_casts_keep_their_own_checks():
    """With `picks` (the magician's) nothing here: `refused` says what her casts go at."""
    a, _ = _attacks(picks=True)
    assert a.worth_a_throw(_monster("horny-mushroom", 120.0))
    h = SimpleNamespace(attacks=a)
    missed = [_monster("horny-mushroom", 120.0)]
    assert fight.shots_at(h, _look([], time.monotonic()), missed, []) == (missed, [])


# --- what she shoots and what she keeps off ---------------------------------------------------------------------------


def _map():
    return build([(0, 500, 1000, 500), (0, 300, 1000, 300)], [("19", 800, 500)])


def _on_her(model, x):
    return Monster(Mob("horny-mushroom", 0, 0, 30, 30, 0.9), x, 500.0, 1, track=5)


def test_one_on_her_body_is_kept_off_though_no_throw_is_worth_it():
    """Positioning isn't attacking: a monster the look missed that no throw is worth (stance's `unsure`), too close to
    her, has her step or jump away as one she'd throw at would; no throw at it, none but a jump back's, whose key goes
    by what's there (behavior.attacks); with nothing in `unsure`, she stays."""
    model = _map()
    here = model.platform(1)
    danger = Danger(model, HER)
    m = _on_her(model, 520.0)
    act = stance(500.0, 500.0, here, [], True, RIGHT, tops={"horny-mushroom": 51.0}, halves={"horny-mushroom": 28.0},
                 around=[m], danger=danger, unsure=[m], moves=0)
    assert act[0] in ("back", "jump_back")
    assert act[0] != "back" or act[1] == LEFT  # (away from it)
    assert stance(500.0, 500.0, here, [], True, RIGHT, tops={"horny-mushroom": 51.0}, halves={"horny-mushroom": 28.0},
                  around=[m], danger=danger, moves=0)[0] == "wait"
    # ... and a monster she does shoot in reach goes as ever, with it far from her
    far = _on_her(model, 900.0)
    shot = Monster(Mob("horny-mushroom", 0, 0, 30, 30, 0.9), 300.0, 500.0, 1, track=6)
    assert stance(500.0, 500.0, here, [shot], True, RIGHT, tops={"horny-mushroom": 51.0},
                  halves={"horny-mushroom": 28.0}, around=[far, shot], danger=danger, unsure=[far],
                  moves=0)[0] == "throw"


def test_shots_at_splits_what_she_shoots_from_what_her_body_keeps_off():
    model = _map()
    now = time.monotonic()
    a, _ = _attacks()
    good = _thrown_at_then_gone(a, ago=0.2)
    dead = _monster("horny-mushroom", 150.0, None, 11)
    a._last.tracks.append(_track(11, 150.0, now - 1.6, HIDDEN, seen=2))
    h = SimpleNamespace(attacks=a, danger=Danger(model, HER), max_hp=None)
    shot, left = fight.shots_at(h, a._last, [good, dead], [])
    assert shot == [good] and left == [dead]  # (before her max HP is read, kept off)
    h.max_hp, h.exp_rate, h.report = 1200.0, RATE, SimpleNamespace(touches=10, misses=3)
    a._last.tracks[-1].blind = 9
    shot, left = fight.shots_at(h, a._last, [good, dead], [])
    assert shot == [good] and left == []  # (hardly a chance it's there: not worth a step)


def test_her_body_doesnt_keep_off_what_counts_as_no_danger():
    model = build([(0, 500, 1000, 500)], [("7", 800, 500)])
    a, _ = _attacks()
    h = SimpleNamespace(attacks=a, danger=Danger(model, HER), max_hp=None)
    a.on_look(_look([], time.monotonic()))
    assert not fight._not_thrown_at(h, a, _monster("slime", 100.0))


def test_one_track_twice_is_one():
    """2026-11-04 00:25:14.839: one track counted twice, -52 and -152: the same by track wherever each has it; two tracks
    side by side are two."""
    a, b = _monster("horny-mushroom", -52.0, track=68), _monster("horny-mushroom", -152.0, track=68)
    assert _same(a, b)
    assert not _same(a, _monster("horny-mushroom", -52.0, track=69))
    assert _same(_monster("horny-mushroom", -52.0), _monster("horny-mushroom", -30.0))  # (no feed: by place)
    assert not _same(_monster("horny-mushroom", -52.0), _monster("horny-mushroom", -130.0))


def test_a_stand_in_thrown_at_is_as_old_as_its_sighting():
    """Stamped when she threw, a stand-in stayed hidden HIDDEN_FOR from the throw, 3.3-3.8 s after its sighting."""
    st = FightState()
    m = _monster("horny-mushroom", 100.0, track=5)
    st.aimed = [m]
    st.sightings = {5: 10.0}
    st.note_throw(11.5)
    assert st.thrown == [[m, 10.0]]
    st.forget_hidden(11.9)
    assert st.thrown == []
    # one the look has: as ever
    other = _monster("horny-mushroom", 100.0, track=6)
    st.aimed = [other]
    st.note_throw(12.0)
    assert st.thrown == [[other, 12.0]]


def test_a_hidden_one_not_worth_a_throw_is_no_target_of_the_fight_but_stays_about_her():
    model = _map()
    a, _ = _attacks()
    a.on_look(_look([], time.monotonic()))
    h = SimpleNamespace(model=model, attacks=a, report=SimpleNamespace(exp_kills=0), ledger=None, danger=Danger(model, HER),
                        max_hp=None, kinds_told=lambda n: [], wounds=None)
    st = FightState()
    s = _look([], time.monotonic(), [])
    s.where.platform = s.platform = model.platform(1)
    old = time.monotonic() - 1.7
    m = _monster("horny-mushroom", 120.0, track=3)
    st.thrown = [[m, old]]
    a._last = s
    a.marks.append(attacks_mod.Mark(m, old, a.reckon(m), blank=1))
    targets, visible, hidden = fight._targets(h, st, s, None, None, True, time.monotonic())
    assert [x.track for x in hidden] == [3]  # (about her, for her body)
    assert targets == []  # (gone from the feed, 1.7 s: no target)
    assert [x.track for x in st.unsure] == [3]
