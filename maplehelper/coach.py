"""On-screen coach: a click-through panel at the top of the screen that narrates each step.

Visible over the game only in windowed or borderless-windowed mode, not exclusive fullscreen.
"""

import queue
import tkinter as tk

from . import win32

BG = "#15171d"
TITLE_FG = "#ffc94d"
STEP_FG = "#ffffff"
RECIPE_FG = "#a9bccf"
# Must stay below 1.0: at exactly 1.0 Tk drops the layered window style that click-through needs.
VISIBLE_ALPHA = 0.96
TOP_MARGIN = 60
MAX_WIDTH = 0.6  # wrap text at this fraction of the screen width
POLL_MS = 30


class Coach:
    """step()/done()/request_quit() are safe to call from any thread; run() owns the main thread."""

    def __init__(self):
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._hide_job: str | None = None
        self._quit = False

        root = self.root = tk.Tk()
        root.title("MapleHelper coach")
        root.overrideredirect(True)
        root.configure(bg=BG)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.0)
        frame = tk.Frame(root, bg=BG, padx=28, pady=16)
        frame.pack()
        wrap = int(root.winfo_screenwidth() * MAX_WIDTH)
        self._title = tk.Label(frame, bg=BG, fg=TITLE_FG, font=("Segoe UI", 13, "bold"))
        self._step = tk.Label(frame, bg=BG, fg=STEP_FG, font=("Segoe UI", 26, "bold"),
                              wraplength=wrap, justify="center")
        self._recipe = tk.Label(frame, bg=BG, fg=RECIPE_FG, font=("Segoe UI", 13),
                                wraplength=wrap, justify="center")
        self._title.pack()
        self._step.pack(pady=(4, 0))
        root.update_idletasks()
        win32.make_click_through(root.winfo_id())
        root.after(POLL_MS, self._poll)

    def step(self, title: str, text: str) -> None:
        self._events.put((title, text, "", None))

    def done(self, title: str, text: str, recipe: list[str], linger_ms: int) -> None:
        """Show the ending plus the whole recipe of steps, then fade out after linger_ms."""
        self._events.put((title, text, "  ·  ".join(recipe), linger_ms))

    def request_quit(self) -> None:
        self._quit = True  # only sets a flag, so it's safe from a signal handler

    def run(self) -> None:
        self.root.mainloop()

    def _poll(self) -> None:
        if self._quit:
            self.root.destroy()
            return
        try:
            while True:
                title, text, recipe, linger_ms = self._events.get_nowait()
                self._show(title, text, recipe)
                if linger_ms is not None:
                    self._hide_job = self.root.after(linger_ms, self._hide)
        except queue.Empty:
            pass
        self.root.after(POLL_MS, self._poll)

    def _show(self, title: str, text: str, recipe: str) -> None:
        if self._hide_job:
            self.root.after_cancel(self._hide_job)
            self._hide_job = None
        self._title.config(text=title.upper())
        self._step.config(text=text)
        if recipe:
            self._recipe.config(text=recipe)
            self._recipe.pack(pady=(8, 0))
        else:
            self._recipe.pack_forget()
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_reqwidth()) // 2
        self.root.geometry(f"+{x}+{TOP_MARGIN}")
        self.root.attributes("-alpha", VISIBLE_ALPHA)

    def _hide(self) -> None:
        self._hide_job = None
        self.root.attributes("-alpha", 0.0)
