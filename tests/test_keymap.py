"""Her keys followed to where the game has them (control.keymap; the user, 2026-10-28: "Follow them", "Remember it"):
`follow` on window maps, the binding remembered and taken at a command's start (her_setup, maplehelper's app), her
moves' and pet food's keys published for the player; her trips' chains taking her keys' names at each chain
(navigator.KeyNames); her moves doubted (keycheck.MoveWatch, what the navigator tells it: _keys_seen); her Lucky Seven's
casts doubted, and out of darts' stop checked first (behavior.hunter.health_check). End to end in the simulator:
loop_tree_grind_jump_key, loop_tree_grind_arrow_key, loop_tree_grind_lucky_seven_key, loop_tree_remembered_keys."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from gamelab import paths
from gamelab.behavior import keycheck
from gamelab.behavior.keycheck import JUMPS_DOUBT, RUNS_DOUBT, Doubt, KeyDoubt, MoveWatch
from gamelab.control import keymap
from gamelab.perception import keybindings as kb
from maplehelper.config import GameKeys
from maplehelper.keys import parse as K

NEVER = keymap.NEVER  # (config.toml's own hotkeys are the same ones today)


def _keys(**kw) -> GameKeys:
    base = dict(left=K("left"), right=K("right"), jump=K("alt"), attack=K("ctrl"), lucky_seven=K("shift"),
                up=K("up"), down=K("down"), pick_up=K("z"), chair=K("d"), sit=K("x"), pet_food=K("u"))
    base.update({k: None if v is None else K(v) for k, v in kw.items()})
    return GameKeys(**base)


def _live(**moved) -> dict:
    """The live window's map (as Reader.functions read it, 2026-10-28 10:56) with `moved` keys' functions instead."""
    names = {**kb.FUNCTIONS, **{f: name for f, (name, _) in kb.PICTURES.items()}}
    out = {k: None for k in kb.KEYS}
    for k, v in kb.LIVE.items():
        out[k] = names.get(v, kb.PICTURE) if v != kb.PICTURE else kb.PICTURE
    out.update(moved)
    return out


@pytest.fixture
def lab(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "LAB", tmp_path)
    monkeypatch.setattr(paths, "ALARM_KEYS", tmp_path / "alarm_keys.json")
    return tmp_path


# --- follow ---------------------------------------------------------------------------------------------------------

def test_the_live_window_follows_nothing():
    f = keymap.follow(_keys(), _live(), never_keys=NEVER)
    assert not f.moved and not f.off and f.stop is None and f.keys == _keys()


@pytest.mark.parametrize("moved, field, key", [
    ({"alt": None, "c": "JUMP"}, "jump", "c"),
    ({"left": None, "q": "MOVE LEFT"}, "left", "q"),
    ({"shift": None, "a": "LUCKY SEVEN"}, "lucky_seven", "a"),
    ({"u": None, "i": "PET FOOD"}, "pet_food", "i"),
    ({"z": None, "a": "PICK UP"}, "pick_up", "a"),
    ({"ctrl": None, "a": "ATTACK"}, "attack", "a"),
    ({"ctrl": "JUMP", "alt": "ATTACK"}, "jump", "ctrl"),   # (swapped: both followed)
])
def test_a_moved_function_is_followed_to_its_key(moved, field, key):
    f = keymap.follow(_keys(), _live(**moved), never_keys=NEVER)
    assert f.moved[field][1] == key and getattr(f.keys, field).name == key and f.stop is None


@pytest.mark.parametrize("moved, stops", [
    ({"alt": None}, "JUMP is on no key"),
    ({"left": None}, "MOVE LEFT is on no key"),
    ({"up": None}, "MOVE UP is on no key"),
    ({"shift": None}, "LUCKY SEVEN is on no key"),
    ({"u": None}, "PET FOOD is on no key"),
    ({"alt": None, "f2": "JUMP"}, "JUMP is on F2 (not a key to press for it)"),     # maplehelper's hotkey
    ({"alt": None, "1": "JUMP"}, "JUMP is on 1 (not a key to press for it)"),       # the alarm's: never a digit
    ({"u": None, "enter": "PET FOOD"}, "PET FOOD is on ENTER (not a key to press"),  # (enter isn't drawn: no key)
])
def test_a_needed_one_on_no_key_she_can_press_stops(moved, stops):
    f = keymap.follow(_keys(), _live(**{k: v for k, v in moved.items() if k in kb.KEYS}), never_keys=NEVER)
    if "ENTER" in stops:  # (the window binds no Enter: her pet food on no key)
        stops = "PET FOOD is on no key"
    assert f.stop is not None and stops in f.stop


def test_her_attack_on_a_digit_is_followed_her_moves_never():
    """The digits, Enter and the rest of alarm.ALARM_UNSAFE bar only what the alarm lets through (her moves and pet
    food): her regular attack may be on 1."""
    f = keymap.follow(_keys(), _live(ctrl=None, **{"1": "ATTACK"}), never_keys=NEVER)
    assert f.keys.attack.name == "1" and f.stop is None


def test_off_and_not_told():
    """Her pick-up on no key: off, said (no stop); her chair's picture not known on its key: kept, not told; her
    pet food dragged (a picture not known on U, none of PET FOOD anywhere): kept; her key's cell illegible: kept."""
    f = keymap.follow(_keys(), _live(z=None), never_keys=NEVER)
    assert f.keys.pick_up is None and "pick_up" in f.off and f.stop is None
    assert any("pick up on no key: not pressed any more" in x for x in f.said)
    f = keymap.follow(_keys(), _live(d=kb.PICTURE), never_keys=NEVER)
    assert f.keys.chair.name == "d" and "chair" in f.unknown
    f = keymap.follow(_keys(), _live(u=kb.PICTURE), never_keys=NEVER)
    assert f.keys.pet_food.name == "u" and "pet_food" in f.unknown and f.stop is None
    f = keymap.follow(_keys(), _live(alt=None), illegible=("alt",), never_keys=NEVER)
    assert f.keys.jump.name == "alt" and "jump" in f.unknown and f.stop is None


def test_one_kept_where_another_lands_is_off():
    """Her HP potion on C (its picture not known: kept), and the game's Lucky Seven moved onto C: her potion's key casts
    now, so it's off, said."""
    keys = _keys(hp_potion="c")
    f = keymap.follow(keys, _live(shift=None, c="LUCKY SEVEN"), never_keys=NEVER)
    assert f.keys.lucky_seven.name == "c" and f.keys.hp_potion is None and "hp_potion" in f.off
    assert any("hp potion C not pressed any more: lucky seven is on it now" in x for x in f.said)


# --- remembered, published ------------------------------------------------------------------------------------------

def test_no_binding_read_config_toml_s(lab):
    start = keymap.at_start(_keys(), "thief")
    assert start.keys == _keys() and start.source == keymap.SOURCE_CONFIG and start.data is None


def test_the_binding_read_is_the_next_start_s_keys(lab):
    """Remembered (her keys as followed, when, off what frame, config.toml's then): the next start's keys, said; config
    .toml's [keys] changed since: said, the remembered still used (`keys --forget` goes back); forgotten: config's."""
    moved = dataclasses.replace(_keys(), jump=K("c"))
    path = keymap.remember(moved, "thief", config_keys=_keys(), frame=lab / "shots" / "keybindings_x.png", preset=2,
                           functions={"c": "JUMP"})
    assert path == lab / "keybindings-thief.json"
    data = json.loads(path.read_text())
    assert data["keys"]["jump"] == "c" and data["config"]["jump"] == "alt" and data["preset"] == 2
    start = keymap.at_start(_keys(), "thief")
    assert start.keys.jump.name == "c" and "JUMP ALT -> C" in start.source and "keybindings_x.png" in start.source
    other = dataclasses.replace(_keys(), pick_up=K("y"))
    start = keymap.at_start(other, "thief")
    assert start.keys.jump.name == "c" and "config.toml [keys] changed since it was read" in start.source
    assert keymap.at_start(_keys(), "magician").source == keymap.SOURCE_CONFIG  # (another character's: her own)
    assert keymap.forget("thief") and not keymap.forget("thief")
    assert keymap.at_start(_keys(), "thief").keys == _keys()


@pytest.mark.parametrize("change, why", [
    ({"jump": "left"}, "two of hers on one key"),
    ({"jump": "nosuch"}, "not a key config.toml can name"),
    ({"jump": None}, "her arrows, jump or attack on no key"),
    ({"jump": "f2"}, "not a key to press for her"),
    ({"pet_food": "1"}, "a key not pressed with the alarm up"),
    ({"wings": "w"}, "unknown fields"),
])
def test_a_remembered_binding_not_as_keys_can_be_is_left_out(lab, change, why):
    data = {"keys": {**keymap.names(_keys()), **change}, "read_at": "x"}
    (lab / "keybindings-thief.json").write_text(json.dumps(data))
    start = keymap.at_start(_keys(), "thief")
    assert start.keys == _keys() and why in start.source and "left out" in start.source


def test_attack_on_no_key_remembered_starts_the_regular_attack_off(lab):
    keymap.remember(_keys(), "thief", attack_off="ATTACK in the tray (on no key)")
    start = keymap.at_start(_keys(), "thief")
    assert start.attack_off == "ATTACK in the tray (on no key)" and start.keys.attack.name == "ctrl"


def test_published_for_the_player_only_her_moves_and_pet_food(lab):
    from gamelab.control.alarm import AlarmKeys
    assert keymap.publish(dataclasses.replace(_keys(), jump=K("c")), why="test")
    data = json.loads(paths.ALARM_KEYS.read_text())
    assert set(data["keys"]) == {"left", "right", "jump", "up", "down", "pet_food"} and data["keys"]["jump"] == "c"
    assert AlarmKeys(_keys(), paths.ALARM_KEYS).current().jump.name == "c"
    assert not keymap.publish(dataclasses.replace(_keys(), jump=K("1")))  # (never a digit: not written)
    assert json.loads(paths.ALARM_KEYS.read_text())["keys"]["jump"] == "c"


def test_her_setup_takes_the_remembered_binding(lab):
    from gamelab.control.timeline import her_setup
    cfg, keys, _ = her_setup(remembered=False)
    keymap.remember(dataclasses.replace(cfg.keys, jump=K("c")), "thief", config_keys=cfg.keys)
    assert her_setup()[1].jump.name == "c" and her_setup(remembered=False)[1] == cfg.keys
    assert her_setup(regular_attack=True)[1].lucky_seven is None


def test_maplehelper_s_app_follows_the_binding_read_too(tmp_path):
    from maplehelper import config
    from maplehelper.combos import COMBOS
    cfg = config.load(paths.CONFIG, (c.name for c in COMBOS))
    path = tmp_path / "keybindings-thief.json"
    assert config.followed(cfg, path) == (cfg, None)
    path.write_text(json.dumps({"keys": {**keymap.names(cfg.keys), "jump": "c"}, "read_at": "2026-10-28 16:00:00"}))
    out, note = config.followed(cfg, path)
    assert out.keys.jump.name == "c" and "jump ALT -> C" in note and "16:00:00" in note
    path.write_text(json.dumps({"keys": {**keymap.names(cfg.keys), "jump": "f1"}}))  # (a combo's button)
    out, note = config.followed(cfg, path)
    assert out is cfg and "left out" in note


# --- her trips' chains, her actions -----------------------------------------------------------------------------------

def test_her_chains_take_her_keys_names_at_each_chain():
    from gamelab.control.actions import Actions
    from gamelab.control.navigator import KeyNames
    from gamelab.motion.flow import Keys
    a = Actions(_keys())
    names = KeyNames(a)
    assert dict(names) == {"left": "left", "right": "right", "jump": "alt", "up": "up", "down": "down", "pick": "z"}
    a.follow(dataclasses.replace(a.keys, jump=K("c"), pick_up=None))
    k = Keys(names)
    k.tap(0.1, "jump")
    assert k.sorted()[0][1] == "c" and "pick" not in names and a.moves_only().keys.jump.name == "c"


# --- her moves doubted ------------------------------------------------------------------------------------------------

def test_her_moves_doubted_jumps_and_runs_in_a_row_once():
    w = MoveWatch()
    for i in range(JUMPS_DOUBT - 1):
        w.jumped(float(i), False)
    w.jumped(9.0, True)  # (one lifted her: the run starts again)
    assert w.due() is None
    for i in range(JUMPS_DOUBT):
        w.jumped(10.0 + i, False)
    d = w.due()
    assert d.kind == "moves" and "HER JUMPS DIDN'T LIFT HER 6 TIMES IN A ROW" in d.text() and w.due() is None
    for i in range(RUNS_DOUBT):
        w.ran(20.0 + i, False)
    assert "HER RUNS DIDN'T MOVE HER 4 TIMES IN A ROW" in w.due().text()


def test_the_navigator_tells_the_watch_what_her_dot_showed_of_a_chain():
    """A chain's jump (a leg of kind jump, an arc leaving upward) and its run (a span of 60 units and more), its glances'
    dots placed on the map: her dot rose 40 over where it was, and went 30 along the run: told so; not, told so; a
    chain cut short before a jump's window is seen through: not told; a climb's grab: not told."""
    from gamelab.control.navigator import Navigator, _Running
    from gamelab.motion.bodies import Arc
    from gamelab.motion.flow import Leg, Span
    calls = []
    nav = object.__new__(Navigator)
    nav.moves = SimpleNamespace(jumped=lambda t, ok: calls.append(("jump", ok)), ran=lambda t, ok: calls.append(("run", ok)))
    nav.senses = SimpleNamespace(localizer=SimpleNamespace(to_map=lambda d: d))
    move = SimpleNamespace(kind="jump")
    leg = Leg(move, None, None, spans=[Span(1, 0.0, 0.5, 0.0, 75.0)], air=[Arc(0.5, 1.0, 75.0, 0.0, 100.0, -80.0, -555.0)])
    running = _Running("p", [leg], 100.0, False)
    moving = [(100.0 + i / 15, (i * 10.0, 0.0 if i < 8 else -60.0)) for i in range(30)]
    nav._keys_seen(running, moving)
    assert calls == [("jump", True), ("run", True)]
    calls.clear()
    nav._keys_seen(running, [(100.0 + i / 15, (0.0, 0.0)) for i in range(30)])
    assert calls == [("jump", False), ("run", False)]
    calls.clear()
    nav._keys_seen(running, [(100.0 + i / 15, (0.0, 0.0)) for i in range(5)], until=100.3)
    assert calls == []  # (cut at 0.3 s: the run's 45 units not gone yet, the jump not seen)
    running = _Running("p", [dataclasses.replace(leg, move=SimpleNamespace(kind="climb"), spans=[])], 100.0, False)
    nav._keys_seen(running, [(100.0 + i / 15, (0.0, 0.0)) for i in range(30)])
    assert calls == []


# --- her health check: the doubts raised ------------------------------------------------------------------------------

def _health_hunter(doubts_raise=True):
    from gamelab.behavior.darts import DartWatch
    from gamelab.behavior.hunter import Hunter
    from gamelab.behavior.keycheck import RegularWatch
    h = object.__new__(Hunter)
    h.hp_watch = SimpleNamespace(due=lambda: None)
    h.darts, h.moves, h.doubts_raise, h.lines = DartWatch(), MoveWatch(), doubts_raise, []
    h.attacks = SimpleNamespace(watch=RegularWatch(), casts=None)
    h.log, h.drink_if_low, h._messages = h.lines.append, lambda s: None, lambda: ""
    h._hear_chat = lambda s: None
    return h


def _out_of_darts(h, t=100.0):
    h.darts.keys = [t + i for i in range(h.darts.presses)]
    h.darts.shown = h.darts.presses


def test_out_of_darts_due_in_the_hunt_loop_checks_the_key_bindings_first(monkeypatch):
    """Her Lucky Seven presses took no MP 20 times: in the hunt loop KeyDoubt (a stars doubt holding the stop), the
    check first (her Lucky Seven's key may be another); outside it, or with the game's word that she's out, or within
    a check's gap: the stop at once, as before."""
    from gamelab.behavior import hunter as hunter_mod
    from gamelab.behavior.darts import OutOfDarts
    monkeypatch.setattr(hunter_mod.client, "check_alarm", lambda: None)
    h = _health_hunter()
    _out_of_darts(h)
    with pytest.raises(KeyDoubt) as e:
        h.health_check(SimpleNamespace(t=200.0, hp=1.0))
    assert e.value.doubt.kind == "stars" and isinstance(e.value.doubt.stop, OutOfDarts)
    h = _health_hunter(doubts_raise=False)
    _out_of_darts(h)
    with pytest.raises(OutOfDarts):
        h.health_check(SimpleNamespace(t=200.0, hp=1.0))
    h = _health_hunter()
    _out_of_darts(h)
    h.darts.told = "You have no throwing stars"
    with pytest.raises(OutOfDarts):
        h.health_check(SimpleNamespace(t=200.0, hp=1.0))


def test_a_stars_check_put_off_waits_stars_wait_s_at_the_most_then_the_stop(monkeypatch):
    from gamelab.behavior import hunter as hunter_mod
    from gamelab.behavior.darts import OutOfDarts
    stop = OutOfDarts(20, 24.0, 1.0)
    h = _health_hunter()
    now = [1000.0]
    monkeypatch.setattr(hunter_mod.time, "monotonic", lambda: now[0])
    h.attacks.watch.deferred = Doubt.stars(stop, now[0])
    h.attacks.watch.skill_held = True
    now[0] += hunter_mod.STARS_WAIT_S - 1
    h.health_check(SimpleNamespace(t=now[0], hp=1.0, where=None))
    now[0] += 2
    with pytest.raises(OutOfDarts) as e:
        h.health_check(SimpleNamespace(t=now[0], hp=1.0, where=None))
    assert e.value.at == now[0] and not h.attacks.watch.skill_held


def test_her_moves_doubt_in_the_hunt_loop_raises_elsewhere_pages(monkeypatch):
    from gamelab.control import alarm
    pages = []
    monkeypatch.setattr(alarm, "page", lambda title, text: pages.append(title))
    monkeypatch.setattr(keycheck.telemetry, "emit", lambda *a, **k: None)
    for raises in (True, False):
        h = _health_hunter(doubts_raise=raises)
        for i in range(JUMPS_DOUBT):
            h.moves.jumped(float(i), False)
        if raises:
            with pytest.raises(KeyDoubt) as e:
                h.health_check(SimpleNamespace(t=10.0, hp=1.0, where=None))
            assert e.value.doubt.kind == "moves" and not h.attacks.watch.held
        else:
            h.health_check(SimpleNamespace(t=10.0, hp=1.0, where=None))
            assert pages == ["MapleStory: her keys"] and not h.attacks.watch.held
