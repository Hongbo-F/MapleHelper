"""The Lie Detector alarm: once the player has seen it, gamelab refuses to run until a person has
answered it and cleared the alarm (the alarm file, paths.ALARM). Paging whoever is at the PC.
(Spotting it on screen: perception.lie_detector.)

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


def raise_alarm(frame: np.ndarray | None, reason: str = "lie_detector") -> dict:
    """Record the alarm (gamelab refuses to run until it's cleared) and page whoever is at the PC."""
    ALARM.parent.mkdir(parents=True, exist_ok=True)
    info = {"reason": reason, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write(info)  # first: everything else may fail
    if frame is not None:
        shot = ALARM.parent / f"lie_detector_{time.strftime('%H%M%S')}.png"
        try:
            Image.fromarray(frame).save(shot)
            info["screenshot"] = str(shot)
            _write(info)
        except OSError:
            pass
    page("MapleStory Lie Detector on screen", "gamelab stopped all input. Answer it in the game yourself.")
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
