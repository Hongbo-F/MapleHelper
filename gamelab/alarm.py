"""Spotting MapleStory's Lie Detector (its anti-bot check) on screen, so gamelab stops at once.

It comes in more than one form, and each is recognized:
- a countdown window titled "LIE DETECTOR" in yellow letters; its title bar is see-through, so the
  yellow letter shapes are matched rather than raw pixels;
- "The Lie Detector Mini-game" ("Use the mouse to follow the fading figure"): a window with the
  Maple Administrator's portrait and a blue header, matched at half size. On 2026-10-18 this one
  wasn't recognized yet, went unanswered, and she was sent to the Truth Booth;
- the Truth Booth itself (where a failed test sends her), by its "Time Left" timer.
Never try to answer it automatically: a person has to.
"""

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .paths import ALARM

ASSETS = Path(__file__).resolve().parent / "assets"
TITLE = ASSETS / "lie_detector_title.png"
THRESHOLD = 0.6  # yellow-letter match; the real window scores ~1.0, game scenes stay well below
# Half-size grey templates and their thresholds. The real screens score 1.0; across ~190 saved game
# screenshots and recordings nothing else scored above 0.53 (mini-game) or 0.69 (timer).
HALF_SIZE = {
    "mini-game": (ASSETS / "lie_detector_minigame.png", 0.8),
    "truth booth": (ASSETS / "truth_booth_timer.png", 0.85),
}


def yellow(rgb: np.ndarray) -> np.ndarray:
    """1.0 where a pixel is the Lie Detector title's yellow, else 0.0."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return ((r > 180) & (g > 140) & (b < 110) & (r - b > 110)).astype(np.float32)


class LieDetectorSpotter:
    """Works on full-size (unscaled) RGB frames of the game's client area."""

    def __init__(self, title_png: Path = TITLE, threshold: float = THRESHOLD):
        self.template = yellow(np.asarray(Image.open(title_png).convert("RGB")))
        self.threshold = threshold
        self.half = {name: (cv2.cvtColor(np.asarray(Image.open(png).convert("RGB")), cv2.COLOR_RGB2GRAY), t)
                     for name, (png, t) in HALF_SIZE.items()}

    def scores(self, frame: np.ndarray) -> dict[str, float]:
        """How well each form of the Lie Detector matches, as a fraction of its threshold (>= 1: seen)."""
        out = {"countdown": self._yellow_title(frame) / self.threshold}
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(cv2.resize(np.ascontiguousarray(frame), (w // 2, h // 2), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_RGB2GRAY)
        for name, (template, threshold) in self.half.items():
            if gray.shape[0] >= template.shape[0] and gray.shape[1] >= template.shape[1]:
                score = float(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED).max())
                out[name] = score / threshold
        return out

    def score(self, frame: np.ndarray) -> float:
        """The best match over all forms, scaled so that `threshold` means seen."""
        return max(self.scores(frame).values()) * self.threshold

    def _yellow_title(self, frame: np.ndarray) -> float:
        mask = yellow(frame)
        if mask.sum() < 0.5 * self.template.sum():  # not even enough yellow for the title
            return 0.0
        return float(cv2.matchTemplate(mask, self.template, cv2.TM_CCOEFF_NORMED).max())

    def seen(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.threshold


def raise_alarm(frame: np.ndarray | None, reason: str = "lie_detector") -> dict:
    """Record the alarm (gamelab refuses to run until it's cleared) and page whoever is at the PC."""
    ALARM.parent.mkdir(parents=True, exist_ok=True)
    info = {"reason": reason, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if frame is not None:
        shot = ALARM.parent / f"lie_detector_{time.strftime('%H%M%S')}.png"
        Image.fromarray(frame).save(shot)
        info["screenshot"] = str(shot)
    ALARM.write_text(json.dumps(info, indent=1), encoding="utf-8")
    page("MapleStory Lie Detector on screen", "gamelab stopped all input. Answer it in the game yourself.")
    return info


def clear_alarm() -> dict | None:
    """Remove the alarm (after a person has dealt with the Lie Detector); returns what it was."""
    info = alarm_info()
    ALARM.unlink(missing_ok=True)
    return info


def alarm_info() -> dict | None:
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
        for _ in range(3):
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            time.sleep(0.4)
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


def _ps(s: str) -> str:
    return s.replace("'", "''")
