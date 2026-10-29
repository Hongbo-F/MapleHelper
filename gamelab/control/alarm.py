"""The Lie Detector alarm: once the player has seen it, gamelab refuses to run until a person has
answered it and cleared the alarm (the alarm file, paths.ALARM). Paging whoever is at the PC.
(Spotting it on screen: perception.lie_detector.) And the two kinds of plan that still go while it's up.

The elevated key player loads this module: it imports nothing else of gamelab but paths.
"""

import json
import subprocess
import time

import numpy as np
from PIL import Image

from ..paths import ALARM

BEEPS = 3         # alert sounds when paging ...
BEEP_GAP_S = 0.4  # ... this far apart
LIE_DETECTOR_ERROR = "Lie Detector on screen: all input stopped"  # the player's answer to a plan it refuses for it
# What's paged for each form the spotter knows (perception.lie_detector, by its names): the notice and the quiz's
# warning come before their test, so the page says how long there is.
PAGED = {
    "mini-game notice": ("MapleStory Lie Detector mini-game in 10 s",
                         "gamelab stopped the hunt (a hunt moves her to a safe place). Follow the figure with the "
                         "mouse yourself."),
    "mini-game": ("MapleStory Lie Detector mini-game on screen",
                  "gamelab stopped all input. Follow the figure with the mouse yourself."),
    "quiz warning": ("MapleStory image quiz in 30 s",
                     "gamelab stopped the hunt (a hunt moves her to a safe place). Click the matching images and "
                     "Confirm yourself."),
    "quiz": ("MapleStory image quiz on screen",
             "gamelab stopped all input. Click the matching images and Confirm yourself."),
}
PAGE = ("MapleStory Lie Detector on screen", "gamelab stopped all input. Answer it in the game yourself.")

# What still goes with the alarm up, named by a plan's `kind` (control.client.send), each only as its kind says
# (control.player.alarm_refusal); every other plan is refused, as before. The user, 2026-10-24, after a Lie Detector
# failed over Chrome Remote Desktop's lag while the alarm kept the pet from being fed: "just stop any in-progress hunt
# when this happens, don't block pet feeding", and a minute later "On a lie detector you should move to a safe place,
# within 10s".
PET_FOOD = "pet_food"  # one tap of her pet food key and nothing else, PET_FOOD_PER_MIN a minute at most
RETREAT = "retreat"    # her moves to a safe place (behavior.retreat), up to RETREAT_WINDOW_S after the sighting
PLAN_KINDS = (PET_FOOD, RETREAT)
PET_FOOD_PER_MIN = 3   # a food is given at most every 75 s or so (behavior.pet: a point of fullness a TICK_S)
RETREAT_S = 10.0       # s from the sighting by which she's on a safe place (behavior.retreat plans for it) ...
RETREAT_WINDOW_S = 12.0  # ... and the player takes her retreat's moves this long after it, then none

# Her keys for those two, as the running commands follow them (the user, 2026-10-28: "Follow them", her moves' keys moved
# in the game; behavior.keycheck, control.keymap): the elevated player, which reads config.toml once at its start,
# learns them from paths.ALARM_KEYS, a file of key names alone that it reads again whenever it changes (`alarm_keys`).
# Only these fields: what goes with the alarm up stays her moves and her pet food, by whatever keys they're on.
ALARM_FIELDS = ("left", "right", "jump", "up", "down", "pet_food")
# Keys none of them is followed onto (control.keymap) nor taken from the file: Esc (it closes or opens the game's windows)
# and the backslash (the KEY BINDINGS window); Enter, Tab and Backspace (a form's: the image quiz has its Confirm, the
# keypad quiz its numbers to type); the digits (the Violetta quiz's buttons are 1 to 4, the keypad's numbers): with the
# alarm up a retreat's move or a pet food tap must never answer the Lie Detector. (maplehelper's hotkeys too: the
# player and control.keymap add config.toml's [hotkeys] and [control].)
ALARM_UNSAFE = ("esc", "\\", "enter", "tab", "backspace", *"0123456789")
ALARM_MOVES = ("left", "right", "jump", "up", "down")


def alarm_keys_problem(names: dict, unsafe=ALARM_UNSAFE) -> str | None:
    """Why key names `names` (ALARM_FIELDS -> a key name, or None: not bound) can't be the alarm's allowance (None: they
    can): a field not one of ALARM_FIELDS, a name config.toml can't name (maplehelper.keys.KEY_CODES), one of `unsafe`,
    two of her moves on one key, her pet food on one of her moves' keys, or her arrows or jump on none (named None)."""
    from maplehelper.keys import KEY_CODES
    if not isinstance(names, dict) or (extra := set(names) - set(ALARM_FIELDS)):
        return f"not only her moves and pet food ({', '.join(sorted(map(str, extra)))})" if isinstance(names, dict) \
            else "not a table of her keys"
    for f, k in names.items():
        if k is None:
            continue
        if not isinstance(k, str) or k not in KEY_CODES:
            return f"{f} {k!r}: not a key config.toml can name"
        if k in unsafe:
            return f"{f} on {k!r}: a key not pressed with the alarm up"
    moves = [names.get(f) for f in ALARM_MOVES if names.get(f) is not None]
    if any(f in names and names[f] is None for f in ("left", "right", "jump")):
        return "her arrows or jump on no key"
    if len(set(moves)) != len(moves):
        return "two of her moves on one key"
    if names.get("pet_food") is not None and names["pet_food"] in moves:
        return "her pet food on one of her moves' keys"
    return None


class AlarmKeys:
    """Her moves' and pet food's keys for the two kinds of plan that go with the alarm up (the player's: `current`):
    config.toml's (`base`, read at the player's start) with paths.ALARM_KEYS's over them, the file read again whenever it
    changes (its size or modification time). A file that can't be read, or doesn't hold only her keys as they can be
    (alarm_keys_problem: each a key config.toml can name, none of `unsafe`, her five moves on five keys, her pet food on
    none of those) is left out: config.toml's then, said once a change of it (`log`). What it holds can only move which
    keys her moves and pet food are on: the player lets through with the alarm up what it always did, a tap of her pet
    food (PET_FOOD_PER_MIN) and her retreat's moves (RETREAT_WINDOW_S), by these keys."""

    def __init__(self, base, path=None, unsafe=ALARM_UNSAFE, log=None):
        from ..paths import ALARM_KEYS
        self.base, self.path = base, ALARM_KEYS if path is None else path
        self.unsafe, self.log = tuple(unsafe), log or (lambda msg: None)
        self._stamp: tuple | None = None
        self._keys = base
        self.why: str | None = None   # why the file was left out (None: used, or not there)
        self.used: dict | None = None  # the file's keys in use (None: config.toml's)

    def current(self):
        """Her keys as the alarm's allowance takes them now (a GameKeys; `base` if None)."""
        if self.base is None:
            return None
        try:
            st = self.path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            stamp = None
        except OSError as e:
            stamp = ("error", str(e))
        if stamp == self._stamp:
            return self._keys
        self._stamp = stamp
        keys, why, used = self.base, None, None
        if stamp is not None:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                names = data.get("keys") if isinstance(data, dict) else None
                why = alarm_keys_problem(names, self.unsafe)  # (the file's alone: only her keys, as they can be)
                if why is None:  # (... and with config.toml's where it gives none)
                    merged = {f: getattr(getattr(self.base, f, None), "name", None) for f in ALARM_FIELDS}
                    merged.update(names)
                    why = alarm_keys_problem(merged, self.unsafe)
                if why is None:
                    keys, used = self._over(names), merged
            except (OSError, ValueError) as e:
                why = f"not read ({type(e).__name__})"
        if why is not None:
            self.log(f"her keys for the alarm: {self.path.name} left out ({why}): config.toml's")
        elif used is not None and used != self.used:
            self.log(f"her keys for the alarm from {self.path.name}: " + ", ".join(
                f"{f} {used.get(f) or '-'}" for f in ALARM_FIELDS))
        self._keys, self.why, self.used = keys, why, used
        return keys

    def _over(self, names: dict):
        """`base` with the file's keys in place of its own (a field it doesn't give: `base`'s)."""
        import dataclasses

        from maplehelper.keys import parse
        return dataclasses.replace(self.base, **{f: None if k is None else parse(k) for f, k in names.items()})


def raise_alarm(frame: np.ndarray | None, reason: str = "lie_detector", form: str | None = None) -> dict:
    """Record the alarm (gamelab refuses to run until it's cleared) and page whoever is at the PC. An alarm that
    can't be written (the disk full) is still paged, and its error raised after that: it had raised before the
    paging, so then nobody was told. `at` is when (time.time(): her retreat counts its 10 s from it); `form` the form
    seen (perception.lie_detector's name), recorded and paged by name (PAGED)."""
    info = {"reason": reason, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "at": round(time.time(), 3)}
    if form is not None:
        info["form"] = form
    try:
        ALARM.parent.mkdir(parents=True, exist_ok=True)
        _write(info)  # first: everything else may fail
        if frame is not None:
            shot = ALARM.parent / f"lie_detector_{time.strftime('%H%M%S')}.png"
            try:
                Image.fromarray(frame).save(shot)
                info["screenshot"] = str(shot)
                _write(info)
            except OSError:
                pass
    finally:
        page(*PAGED.get(form, PAGE))
    return info


def clear_alarm() -> dict | None:
    """Remove the alarm (after a person has dealt with the Lie Detector); returns what it was."""
    info = alarm_info()
    ALARM.unlink(missing_ok=True)
    return info


def alarm_info() -> dict | None:
    """What the alarm says (reason, time, screenshot), or None if no alarm is up."""
    if not ALARM.exists():
        return None
    try:
        return json.loads(ALARM.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"reason": "unknown"}


def page(title: str, text: str) -> None:
    """Get attention at the PC without taking focus from the game: alert sounds plus a Windows
    notification. (Paging a phone is done by whoever drives gamelab, e.g. Claude's push.)"""
    try:
        import winsound
        for _ in range(BEEPS):
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            time.sleep(BEEP_GAP_S)
    except Exception:
        pass
    toast = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null;"
        "$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
        "$n = $t.GetElementsByTagName('text');"
        f"$n.Item(0).AppendChild($t.CreateTextNode('{_ps(title)}')) | Out-Null;"
        f"$n.Item(1).AppendChild($t.CreateTextNode('{_ps(text)}')) | Out-Null;"
        "$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe';"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show("
        "[Windows.UI.Notifications.ToastNotification]::new($t))"
    )
    try:
        subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", toast],
                         creationflags=subprocess.CREATE_NO_WINDOW)
    except OSError:
        pass


def keep_recent(seconds: float, name: str) -> str:
    """Have the rolling recorder save its last `seconds` (control.rolling.ask: lab/recent/<time>-<name>/), as
    `save-recent` does, for a stop whose screen nobody has seen yet (behavior.retreat.on_alarm): what came of it, said
    for the log (its folder, or why nothing was saved). Never raises. (Imported only when it's asked: the player loads
    this module.)"""
    try:
        from . import rolling
        out = rolling.ask(seconds, name)
    except Exception as e:  # (no recorder running, or it didn't answer)
        return f"the recording not saved ({type(e).__name__}: {e})"
    if "error" in out:
        return f"the recording not saved: {out['error']}"
    return f"the recording's last {out.get('seconds', seconds):g} s saved: {out.get('folder')}"


def _write(info: dict) -> None:
    ALARM.write_text(json.dumps(info, indent=1), encoding="utf-8")


def _ps(s: str) -> str:
    """`s` quoted for a single-quoted PowerShell string."""
    return s.replace("'", "''")
