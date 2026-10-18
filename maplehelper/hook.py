"""System-wide keyboard hook: sees every key she presses and can hide some from the game."""

import ctypes
import threading
import traceback
from collections.abc import Callable
from ctypes import wintypes

from . import win32
from .win32 import kernel32, user32


class KeyboardHook:
    """Runs a WH_KEYBOARD_LL hook on its own thread.

    on_key(vk, is_down) is called for every real key event (our own injected keys are skipped)
    and returns True to swallow the event so no other program sees it. It runs inside the hook,
    so it must return quickly: Windows silently removes hooks that are too slow.
    """

    def __init__(self, on_key: Callable[[int, bool], bool]):
        self._on_key = on_key
        self._proc = win32.HOOKPROC(self._callback)  # must stay referenced while the hook is installed
        self._thread = threading.Thread(target=self._run, name="keyboard-hook", daemon=True)
        self._thread_id = 0
        self._ready = threading.Event()
        self._error: OSError | None = None

    def start(self) -> None:
        self._thread.start()
        self._ready.wait()
        if self._error:
            raise self._error

    def stop(self) -> None:
        if self._thread.is_alive():
            user32.PostThreadMessageW(self._thread_id, win32.WM_QUIT, 0, 0)
            self._thread.join(timeout=2)

    def _run(self) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()
        hook = user32.SetWindowsHookExW(win32.WH_KEYBOARD_LL, self._proc, kernel32.GetModuleHandleW(None), 0)
        if not hook:
            self._error = ctypes.WinError(ctypes.get_last_error())
            self._ready.set()
            return
        self._ready.set()
        msg = wintypes.MSG()
        try:
            # The hook callback is delivered while this thread waits for messages.
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                pass
        finally:
            user32.UnhookWindowsHookEx(hook)

    def _callback(self, n_code: int, w_param: int, l_param: int) -> int:
        if n_code == win32.HC_ACTION:
            event = ctypes.cast(l_param, ctypes.POINTER(win32.KBDLLHOOKSTRUCT)).contents
            if event.dwExtraInfo != win32.INJECTED_TAG:
                is_down = w_param in (win32.WM_KEYDOWN, win32.WM_SYSKEYDOWN)
                try:
                    if self._on_key(event.vkCode, is_down):
                        return 1
                except Exception:
                    traceback.print_exc()  # never let a bug here break her keyboard
        return user32.CallNextHookEx(None, n_code, w_param, l_param)
