"""A stand-in game window for trying combos without MapleStory: lists every key it receives, with timing.

    python -m maplehelper.keyviewer [--log FILE]

Its title contains "MapleStory", so MapleHelper's hotkeys work while it is focused.
"""

import argparse
import time
import tkinter as tk

TITLE = "MapleStory Key Viewer"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m maplehelper.keyviewer", description=__doc__.splitlines()[0])
    parser.add_argument("--log", help="also write each key event to this file")
    args = parser.parse_args(argv)
    log_file = open(args.log, "w", encoding="utf-8", buffering=1) if args.log else None

    root = tk.Tk()
    root.title(TITLE)
    root.geometry("560x460")
    held_text = tk.StringVar(value="held: (none)")
    tk.Label(root, textvariable=held_text, font=("Consolas", 12), anchor="w").pack(fill="x", padx=8, pady=(8, 0))
    view = tk.Text(root, font=("Consolas", 11), takefocus=0)
    view.pack(fill="both", expand=True, padx=8, pady=8)
    view.insert("end", "Focus this window and press a combo hotkey.\n\n")
    view.configure(state="disabled")

    start = last = 0.0
    held: set[str] = set()

    def record(event: tk.Event, kind: str) -> str:
        nonlocal start, last
        now = time.perf_counter()
        name = event.keysym
        if kind == "down":
            if name in held:
                return "break"  # auto-repeat
            held.add(name)
        else:
            held.discard(name)
        if now - last > 2.5:
            start = now  # restart the clock for each new burst of keys
            view.configure(state="normal")
            view.insert("end", "\n")
        last = now
        line = f"{(now - start) * 1000:7.0f} ms  {kind:<4}  {name}"
        view.configure(state="normal")
        view.insert("end", line + "\n")
        view.see("end")
        view.configure(state="disabled")
        held_text.set("held: " + (" ".join(sorted(held)) or "(none)"))
        if log_file:
            log_file.write(line + "\n")
        return "break"  # keep Alt/F10 from reaching Tk's menu handling

    root.bind("<KeyPress>", lambda e: record(e, "down"))
    root.bind("<KeyRelease>", lambda e: record(e, "up"))
    root.after(200, root.focus_force)
    root.mainloop()


if __name__ == "__main__":
    main()
