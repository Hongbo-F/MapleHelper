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


def _write(info: dict) -> None:
    ALARM.write_text(json.dumps(info, indent=1), encoding="utf-8")


def _ps(s: str) -> str:
    """`s` quoted for a single-quoted PowerShell string."""
    return s.replace("'", "''")
