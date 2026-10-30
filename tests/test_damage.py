"""Her attacks' damage as the monsters' bars show it, learned as she hunts and kept between runs (behavior.damage; the
user, 2026-10-29: "The fitted damage should be dynamically updated as grinding goes, and memorized between runs"), and
her wounds finished first with the imminent danger before them (attack.stance.WOUNDS_FIRST; the user: "yes we should
finish a monster, but we should also watch out for inmennent danger from a close by mob")."""

import random
import time

from maplehelper.engine import LEFT, RIGHT

from gamelab.attack import magic
from gamelab.attack.choice import Attack
from gamelab.behavior import damage
from gamelab.behavior.attacks import Attacks
from gamelab.behavior.damage import AT_LEAST, EXACT, MISS, Book, Read, fit
from gamelab.behavior.ledger import Kill

from test_choice import _look
from test_magician_hunt import MAGE, TREE, _cast, _close_scene, _her_stance


def _casts(lo: int, hi: int, n: int, seed: int = 1) -> list[int]:
    """`n` casts' damage of two hits of lo-hi each, as likely each."""
    rng = random.Random(seed)
    return [rng.randint(lo, hi) + rng.randint(lo, hi) for _ in range(n)]


def _reads(values, kind="horny-mushroom", how=EXACT, tol=8.8, level=None, barred=False, seed=2):
    rng = random.Random(seed)
    return [Read(1000.0 + i, kind, how, v + (rng.uniform(-tol, tol) if how == EXACT else 0.0), tol, level, barred)
            for i, v in enumerate(values)]


def test_the_fit_finds_a_hits_least_and_most_from_its_casts_bars():
    """The night's casts as the bars read them: two hits of 127-149 each give a fit within a few of them, however many
    of the kills of a Green Mushroom (at least 303, its 233 HP before its MDD 30) come with the bars of those it
    survived, and not the profile's prior (76-267, the v83 formula: a Green killed by 69% of the casts that land)."""
    lo, hi = fit(_reads(_casts(127, 149, 120)), 2)
    assert abs(lo - 127) <= 5 and abs(hi - 149) <= 5, (lo, hi)
    # A Green's: the casts under its 303 seen on its bar, those over it kills of 'at least' 303 less a px
    greens = _casts(127, 149, 200, seed=3)
    reads = (_reads([d for d in greens if d < 303], kind="green-mushroom", tol=6.6)
             + _reads([303 - 6.6 for d in greens if d >= 303], kind="green-mushroom", how=AT_LEAST))
    lo, hi = fit(reads, 2)
    assert abs(lo - 127) <= 6 and abs(hi - 149) <= 6, (lo, hi)


def test_too_few_bars_no_fit_and_a_kill_past_any_bar_isnt_taken_as_hers():
    assert fit(_reads(_casts(127, 149, damage.MIN_EXACT - 1)), 2) is None
    # Kills credited to a first cast at a Horny Mushroom's whole 403 (a second cast's, or another player's), a third of
    # the reads: not hers, the fit as without them (they widened it to 76-264 a hit before KILL_SLACK)
    exact = _reads(_casts(127, 149, 100))
    lo, hi = fit(exact + _reads([403 - 8.8] * 50, how=AT_LEAST), 2)
    assert abs(lo - 127) <= 5 and abs(hi - 149) <= 5, (lo, hi)


def test_the_book_keeps_its_reads_and_fit_between_runs_and_her_attack_follows(tmp_path):
    path = tmp_path / "damage.json"
    said = []
    book = Book(path, log=said.append)
    book.threaded = False
    prior = Attack("magic_claw", 2, 76, 267, 20.0)
    assert book.attack(prior) is prior and book.said("magic_claw") == ""
    for r in _reads(_casts(127, 149, 40)):
        book.add("magic_claw", 2, r)
        book._fit_at.clear()  # (each read refits: REFIT_S aside)
    claw = book.attack(prior)
    assert abs(claw.lo - 127) <= 6 and abs(claw.hi - 149) <= 6 and claw.mp == 20.0 and claw.key == "magic_claw"
    assert book.version >= 1 and any("her magic_claw's damage as her throws showed it" in line for line in said)
    book.save()
    again = Book(path, log=lambda line: None)  # (the next run's)
    assert again.attack(prior) == claw and len(again.reads("magic_claw")) == 40
    assert again.said("magic_claw").startswith(f"magic_claw: {claw.lo}-{claw.hi} a hit as learned")
    assert not path.with_name(path.name + ".tmp").exists()
    path.write_text("{not json", encoding="utf-8")  # (a broken file: nothing learned, never the hunt's end)
    assert Book(path, log=lambda line: None).attack(prior) is prior


def test_the_newest_reads_of_her_level_and_the_one_under_it_are_used(tmp_path):
    level = [20]
    book = Book(tmp_path / "damage.json", log=lambda line: None, level=lambda: level[0])
    book.threaded = False
    for r in _reads(_casts(100, 120, 30), level=18) + _reads(_casts(127, 149, 30), level=20):
        book.add("magic_claw", 2, r)
        book._fit_at.clear()
    assert len(book.reads("magic_claw")) == 30  # (level 18's out at 20)
    assert abs(book.attack(Attack("magic_claw", 2, 76, 267, 20.0)).lo - 127) <= 6
    for r in _reads(_casts(127, 149, damage.KEEP + 10)):
        book.add("magic_claw", 2, r)
    assert len(book.data["magic_claw"]["reads"]) == damage.KEEP


def test_her_misses_share_from_the_throws_judged_after_a_bar(tmp_path):
    book = Book(tmp_path / "damage.json", log=lambda line: None)
    book.threaded = False
    assert book.miss("magic_claw", 0.275) == 0.275
    for r in _reads(_casts(127, 149, 18), barred=True) + _reads(_casts(127, 149, 30)):  # (30 from no bar: not judged)
        book.add("magic_claw", 2, r)
    for r in _reads([0.0] * 2, how=MISS, barred=True):
        book.add("magic_claw", 2, r)
    assert book.miss("magic_claw", 0.275) == 2 / 20


def _learning(tmp_path, kills=None, prior=(76, 267)):
    said = []
    book = Book(tmp_path / "damage.json", log=said.append, level=lambda: 29)
    book.threaded = False
    claw = MAGE.attacks["skill"]
    a = Attacks(TREE, MAGE.keys, "skill", max_mp=MAGE.stats.mp, level=0, log=said.append,
                kills=(lambda: kills) if kills is not None else None,
                skill=Attack("magic_claw", claw.hits, prior[0], prior[1], claw.mp), magic=True, no_regular=True,
                picks=True, book=book, her_level=lambda: 29)
    return a, book, said


def test_her_casts_bars_and_kills_teach_the_book_and_her_reckoning_follows(tmp_path):
    """A cast at a whole Horny Mushroom (403, MDD 0), its bar 33% a look 0.7 s after: 270 learned; her second, which
    killed it (the ledger's kill placed on it): at least the 133 it had left. With MIN_EXACT bars learned her Magic Claw
    is the fit's, and after one cast at a Green Mushroom (303 before its MDD) she reckons it alive, a second cast due."""
    kills = []
    with magic.swapped(MAGE):
        a, book, said = _learning(tmp_path, kills)
        t = time.monotonic()
        a.on_look(_look([("horny-mushroom", 200.0, None)], t=t))
        assert _cast(a, at=t + 0.05)
        a.on_look(_look([("horny-mushroom", 195.0, 0.33)], t=t + 0.75))
        (r,) = book.reads("magic_claw")
        assert r.how == EXACT and abs(r.value - 0.67 * 403) < 1 and not r.barred and r.level == 29
        assert _cast(a, at=t + 0.9)
        kills.append(Kill(t + 1.5, "horny-mushroom", 1, 195.0, 0.0, 1, gained=t + 1.4))
        a.on_look(_look([], t=t + 1.5))
        r = book.reads("magic_claw")[-1]
        assert r.how == AT_LEAST and r.barred and abs(r.value - (0.33 * 403 - 403 / 46)) < 1
        # Bars enough: her Magic Claw the fit's at the next look
        for v in _reads(_casts(127, 149, damage.MIN_EXACT + 5)):
            book.add("magic_claw", 2, v)
        book._fit_at.clear()
        book.add("magic_claw", 2, _reads([280.0])[0])
        a.on_look(_look([], t=t + 2.0))
        assert abs(a.skill.lo - 127) <= 8 and abs(a.skill.hi - 149) <= 8, a.skill
        a.on_look(_look([("green-mushroom", 180.0, None)], t=t + 2.5))
        assert _cast(a, at=t + 2.55)
        a.on_look(_look([("green-mushroom", 176.0, None)], t=t + 2.85))  # (before its damage or EXP shows)
        left = a.reckon(a._last.monsters[0])
        assert left.dead < 0.1 and a.refused(a._last.monsters[0]) is None  # (the prior had it 50% dead: refused)


def test_nothing_learned_with_two_casts_between_bars_nor_another_player_near(tmp_path):
    with magic.swapped(MAGE):
        a, book, _ = _learning(tmp_path)
        t = time.monotonic()
        a.on_look(_look([("horny-mushroom", 200.0, None)], t=t))
        assert _cast(a, at=t + 0.05)
        a.on_look(_look([("horny-mushroom", 198.0, None)], t=t + 0.3))
        assert _cast(a, at=t + 0.35)
        a.on_look(_look([("horny-mushroom", 195.0, 0.2)], t=t + 1.2))  # (two casts' damage: which did what?)
        assert book.reads("magic_claw") == []
        assert _cast(a, at=t + 1.25)
        s = _look([("horny-mushroom", 190.0, 0.05)], t=t + 2.0)
        s.players = [type("P", (), {"x": 300.0, "y": 0.0, "platform": 1})()]  # (another player 110 off it)
        a.on_look(s)
        assert book.reads("magic_claw") == []
        assert _cast(a, at=t + 2.05)
        a.on_look(_look([("horny-mushroom", 188.0, 0.05)], t=t + 2.8))  # (its bar as before: her cast took nothing)
        (r,) = book.reads("magic_claw")
        assert r.how == MISS and r.barred


# --- her wounds first, the imminent danger before them -----------------------------------------------------------

def test_she_finishes_the_one_she_hurt_before_a_fresh_one_the_other_way():
    """Live at 20:50:08 a Green Mushroom at 13% 224 to her left was left for a whole Horny Mushroom 174 to her right
    (TOUCHER_FIRST: the Horny on her platform), and the Green walked at her. With `hurt` (her wounds) the cast goes the
    Green's way; without (the thief's), as before."""
    from gamelab.attack import stance
    with magic.swapped(MAGE):
        model, p, mine = _close_scene(500, [("green-mushroom", 276.0), ("horny-mushroom", 674.0)])
        green, horny = mine
        assert _her_stance(model, p, 500, mine) == ("throw", RIGHT)  # (as before: the way she faces)
        act = stance.stance(500, 500, p, mine, True, RIGHT, around=mine,
                            hurt=lambda m: 0.13 if m is green else None)
        assert act == ("throw", LEFT)
        # Two hurt: the one with less left
        act = stance.stance(500, 500, p, mine, True, LEFT, around=mine,
                            hurt=lambda m: 0.13 if m is horny else 0.5)
        assert act == ("throw", RIGHT)


def test_a_mob_about_to_be_too_close_comes_before_her_wound():
    """One on her platform walking at her that will be inside her search's reach by the attack after her next
    (_imminent: IMMINENT_S) is cast at first; one farther off or standing, her wound first."""
    from gamelab.attack import stance
    with magic.swapped(MAGE):
        model, p, mine = _close_scene(500, [("green-mushroom", 300.0), ("horny-mushroom", 640.0)])
        green, horny = mine
        walking = stance.Pace(walk=lambda m: -100.0 if m is horny else None)  # (at her, 100 units/s)
        act = stance.stance(500, 500, p, mine, True, LEFT, around=mine, pace=walking,
                            hurt=lambda m: 0.1 if m is green else None)
        assert act == ("throw", RIGHT)  # (140 off, at her: within its reach by her attack after next)
        standing = stance.Pace(walk=lambda m: None)
        act = stance.stance(500, 500, p, mine, True, RIGHT, around=mine, pace=standing,
                            hurt=lambda m: 0.1 if m is green else None)
        assert act == ("throw", LEFT)


def test_she_waits_for_her_wound_under_her_hits_effect_rather_than_turn_away():
    """Her wound unseen this look (`hidden_hurt`: under her hit's effect) to her left, in her search's reach there, and
    nothing she casts at that way: a wait, not a cast at a fresh one to her right; with one about to be too close, the
    cast goes."""
    from gamelab.attack import stance
    with magic.swapped(MAGE):
        model, p, mine = _close_scene(500, [("horny-mushroom", 700.0)])
        _, _, (ghost,) = _close_scene(500, [("green-mushroom", 330.0)])
        act = stance.stance(500, 500, p, mine, True, LEFT, around=mine, hurt=lambda m: None, hidden_hurt=[ghost])
        assert act == ("wait",)
        assert stance.stance(500, 500, p, mine, True, LEFT, around=mine)[0] == "throw"  # (as before)
        walking = stance.Pace(walk=lambda m: -100.0)
        act = stance.stance(500, 500, p, mine, True, LEFT, around=mine, pace=walking, hurt=lambda m: None,
                            hidden_hurt=[ghost])
        assert act == ("wait",)  # (200 off at 100 units/s: not too close by her attack after next)
        model, p, mine = _close_scene(500, [("horny-mushroom", 630.0)])
        act = stance.stance(500, 500, p, mine, True, LEFT, around=mine, pace=walking, hurt=lambda m: None,
                            hidden_hurt=[ghost])
        assert act == ("throw", RIGHT)  # (130 off walking at her: about to be too close, not waited past)
