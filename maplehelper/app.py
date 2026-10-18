"""Wires everything together. Run with: python -m maplehelper"""

import argparse
import logging
import logging.handlers
import queue
import signal
from pathlib import Path

from . import config, win32
from .coach import Coach
from .combos import COMBOS
from .controller import Controller
from .engine import Runner
from .hook import KeyboardHook

log = logging.getLogger("maplehelper")

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.toml"


class Narrator:
    """Shows each step on the coach overlay and in the console."""

    def __init__(self, coach: Coach):
        self._coach = coach

    def step(self, title: str, text: str) -> None:
        log.info("%s | %s", title, text)
        self._coach.step(title, text)

    def done(self, title: str, text: str, recipe: list[str], linger_ms: int) -> None:
        log.info("%s | %s", title, text)
        self._coach.done(title, text, recipe, linger_ms)


def _start_logging(debug: bool) -> logging.handlers.QueueListener:
    # Log through a queue: printing straight from the keyboard hook could stall it (e.g. while
    # text is selected in the console), and Windows removes hooks that stall.
    q: queue.SimpleQueue = queue.SimpleQueue()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d  %(message)s", "%H:%M:%S"))
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        handlers=[logging.handlers.QueueHandler(q)])
    listener = logging.handlers.QueueListener(q, handler)
    listener.start()
    return listener


def _print_banner(cfg: config.Config) -> None:
    titles = {c.name: c.title for c in COMBOS}
    print(f'MapleHelper is running. Hotkeys work only while the focused window title contains "{cfg.game_window_title}".')
    for name, key in cfg.hotkeys.items():
        print(f"  {key.label:<6} {titles[name]}")
    print(f"  {cfg.abort.label:<6} stop the running combo")
    print(f"  {cfg.slow_mode.label:<6} slow / demo mode on-off")
    print("Press Ctrl+C here to quit.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m maplehelper",
                                     description="One-button escape combos with an on-screen coach.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to config.toml")
    parser.add_argument("--debug", action="store_true",
                        help="log every key press with the focused window's title")
    args = parser.parse_args(argv)

    try:
        cfg = config.load(args.config, (c.name for c in COMBOS))
    except config.ConfigError as e:
        print(f"Config problem: {e}")
        return 2

    listener = _start_logging(args.debug)
    win32.enable_dpi_awareness()
    coach = Coach()
    runner = Runner(win32.send_key, cfg.keys, cfg.timing, Narrator(coach), COMBOS)
    hook = KeyboardHook(Controller(cfg, runner, win32.foreground_title).on_key)
    signal.signal(signal.SIGINT, lambda *_: coach.request_quit())

    with win32.precise_timers():
        runner.start()
        hook.start()
        _print_banner(cfg)
        try:
            coach.run()
        finally:
            hook.stop()
            runner.stop()
            listener.stop()
    return 0
