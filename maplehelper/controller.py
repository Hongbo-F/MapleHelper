"""Decides what each real key press means: run a combo, stop, toggle slow mode, or let it through."""

import logging
from collections.abc import Callable

from . import keys
from .config import Config
from .engine import DROPPED, LEFT, QUEUED, RIGHT, Runner

log = logging.getLogger(__name__)


class Controller:
    def __init__(self, cfg: Config, runner: Runner, foreground_title: Callable[[], str]):
        self._runner = runner
        self._foreground_title = foreground_title
        self._game_title = cfg.game_window_title
        self._triggers = {key.vk: name for name, key in cfg.hotkeys.items()}
        self._left = cfg.keys.left.vk
        self._right = cfg.keys.right.vk
        self._abort = cfg.abort.vk
        self._slow = cfg.slow_mode.vk
        self._held: set[int] = set()  # buttons we swallowed that are still physically down

    def on_key(self, vk: int, is_down: bool) -> bool:
        """Called from the keyboard hook. Returns True to hide the key from the game."""
        if not is_down:
            if vk in self._held:
                self._held.discard(vk)
                return True
            return False
        if vk in self._held:  # keyboard auto-repeat while she holds the button
            return True

        title = self._foreground_title()
        log.debug("key %s down, focused window %r", keys.name_of(vk), title)
        if self._game_title not in title:
            return False

        if vk == self._left:
            self._runner.note_facing(LEFT)
        elif vk == self._right:
            self._runner.note_facing(RIGHT)
        elif vk == self._abort:
            self._runner.abort()
        elif vk == self._slow:
            self._held.add(vk)
            self._runner.toggle_slow()
            return True
        elif vk in self._triggers:
            self._held.add(vk)
            name = self._triggers[vk]
            result = self._runner.trigger(name)
            if result == QUEUED:
                log.info("%s queued: runs when the current combo ends", name)
            elif result == DROPPED:
                log.info("%s ignored: pressed too early in the current combo", name)
            return True
        return False
