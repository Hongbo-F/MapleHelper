"""gamelab command line. Run `python -m gamelab -h`."""

import argparse
import ctypes
import sys
import time

import numpy as np
from PIL import Image

from maplehelper.engine import LEFT, RIGHT
from maplehelper.combos import COMBOS

from . import analyze, calibration, client, window
from .paths import LAB, PLAYER_LOG, RECORDINGS, REPO, STOP
from .timeline import chain, her_setup
from .track import airborne, track


def _game_title() -> str:
    return her_setup()[0].game_window_title


def cmd_player(args) -> int:
    if not args.elevate:
        from .player import serve
        serve()
        return 0
    # ShellExecute "runas" shows the UAC prompt and starts `python -m gamelab player` elevated.
    rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, "-X utf8 -m gamelab player",
                                             str(REPO), 7)  # 7 = minimized, not activated
    if rc <= 32:
        reason = {5: "the UAC prompt was declined", 2: "python not found", 3: "folder not found"}.get(rc, "see ShellExecute docs")
        print(f"couldn't start the player as administrator: error {rc} ({reason})")
        return 1
    print(f"player starting as administrator; log: {PLAYER_LOG}")
    return 0


def cmd_stop(args) -> int:
    LAB.mkdir(parents=True, exist_ok=True)
    STOP.touch()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if PLAYER_LOG.exists() and PLAYER_LOG.read_text(encoding="utf-8").rstrip().endswith("player stopped"):
            print("player stopped")
            return 0
        time.sleep(0.2)
    print("asked the player to stop; it hasn't confirmed (maybe it wasn't running)")
    return 0


def cmd_calibrate(args) -> int:
    window.bring_to_front(_game_title())
    tag = tuple(int(v) for v in args.tag.split(",")) if args.tag else None
    cal = calibration.calibrate(tag)
    print(f"recording crop {cal.crop} (client px), name tag template {cal.template.shape[1]}x{cal.template.shape[0]}")
    print(f"check {LAB / 'calibrate_crop.png'}: her name tag should be near the middle, a bit below centre")
    return 0


def cmd_snap(args) -> int:
    window.bring_to_front(_game_title())
    frame = client.submit(fps=10, duration_ms=200, scale=args.scale, lead_ms=0, name="snap")["frames"][-1]
    path = LAB / "snap.png"
    Image.fromarray(frame).save(path)
    print(path)
    return 0


def cmd_lock(args) -> int:
    """Press an attack, then jump `d` ms later: from when does the game accept the jump again?"""
    cfg, keys, _ = her_setup()
    attack = getattr(keys, args.action)
    if attack is None:
        print(f"{args.action} isn't bound in config.toml")
        return 1
    cal = calibration.load()
    window.bring_to_front(cfg.game_window_title)
    for d in args.delays:
        events = [[0, attack.name, "down"], [40, attack.name, "up"], [d, keys.jump.name, "down"], [d + 40, keys.jump.name, "up"]]
        rec = client.record(events, name=f"lock_{args.action}_{d}", duration_ms=d + 900, crop=cal.crop, scale=cal.scale)
        jumped = [s for s in airborne(track(rec.frames, rec.times, cal.template)) if s[0] >= d]
        print(f"jump {d:4d} ms after {args.action}: " +
              (f"airborne at {jumped[0][0]:.0f} ms" if jumped else "IGNORED (still locked)"))
        time.sleep(0.6)
    return 0


MOVES_BACK = {"jump_back_shot", "hop_back"}  # combos whose jumps must carry her away
FACE_MS = 600  # each run first taps the arrow she should face, like her own last arrow press


def cmd_combo(args) -> int:
    """Play a combo (optionally several back to back) exactly as the helper would, and check each one."""
    cfg, keys, timing = her_setup(regular_attack=args.regular_attack)
    cal = calibration.load()
    window.bring_to_front(cfg.game_window_title)
    good_jump = {"back"} if args.name in MOVES_BACK else {"back", "up", "forward"}
    totals = {"back": 0, "up": 0, "forward": 0, "front": 0, "wrong": 0}
    for run in range(args.runs):
        facing = RIGHT if run % 2 == 0 else LEFT  # alternate so she ends up roughly where she started
        front = keys.right if facing == RIGHT else keys.left
        events, starts, end = chain(args.name, facing, args.chain, keys, timing)
        # Face the way the helper would assume, then run the combo(s) from FACE_MS on.
        events = [[0, front.name, "down"], [timing.turn_tap, front.name, "up"]] + \
                 [[t + FACE_MS, k, kind] for t, k, kind in events]
        starts, end = [s + FACE_MS for s in starts], end + FACE_MS
        rec = client.record(events, name=f"{args.name}_{run}", duration_ms=end + 600, crop=cal.crop, scale=cal.scale)
        print(f"{rec.name} (facing {facing}, {args.chain} in a row):")
        for i, r in enumerate(analyze.combos(rec, starts, end, facing, cal.template)):
            jumps = ", ".join(f"{j.direction} ({j.dx:+.0f})" for j in r.jumps) or "none"
            throws = ", ".join("toward front" if t.toward_front else "WRONG WAY" for t in r.throws) or "none"
            ok = all(j.direction in good_jump for j in r.jumps) and all(t.toward_front for t in r.throws)
            print(f"  #{i} at {r.start - FACE_MS:5d} ms: jumps {jumps}; throws {throws}{'' if ok else '   <--'}")
            for j in r.jumps:
                totals[j.direction] += 1
            totals["front"] += sum(t.toward_front for t in r.throws)
            totals["wrong"] += sum(not t.toward_front for t in r.throws)
        time.sleep(0.6)
    print(f"\njumps: {totals['back']} back, {totals['up']} straight up, {totals['forward']} forward; "
          f"throws: {totals['front']} toward front, {totals['wrong']} wrong way")
    return 0


def cmd_sheet(args) -> int:
    rec = client.load(args.recording)
    keep = [i for i, t in enumerate(rec.times) if args.start <= t <= args.end][::args.every]
    frames, times = rec.frames[keep], rec.times[keep]
    if args.around_her:
        rows = track(frames, times, calibration.load().template)
        crops = []
        for f, (_, x, y, _) in zip(frames, rows):
            c = np.zeros((80, 120, 3), np.uint8)
            src = f[max(0, y - 60):y + 20, max(0, x - 40):x + 80]
            c[:src.shape[0], :src.shape[1]] = src
            crops.append(c)
        frames = np.stack(crops)
    path = LAB / f"{args.recording}_sheet.png"
    client.contact_sheet(frames, times, path, cols=args.cols, every=1, zoom=3 if args.around_her else 1)
    print(path)
    return 0


def cmd_check(args) -> int:
    rec = client.load(args.recording)
    lag = analyze.send_lag(rec)
    print(f"keys: {lag['sent']}/{lag['planned']} sent, late by at most {lag['max_ms']:.1f} ms (mean {lag['mean_ms']:.1f})")
    pace = analyze.frame_pacing(rec)
    print(f"game frames: {pace['frames']}, gap median {pace['median_ms']} ms, p95 {pace['p95_ms']} ms, "
          f"max {pace['max_ms']} ms; gaps over 45 ms (start, length): {pace['long_gaps']}")
    print("(long gaps while she stands still are just frames that didn't change)")
    return 0


def cmd_list(args) -> int:
    for path in sorted(RECORDINGS.glob("*.npz"), key=lambda p: p.stat().st_mtime):
        print(path.stem)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m gamelab", description="Drive the real game with exact key timelines and measure what happened.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("player", help="run the key player (use --elevate: the game runs as admin)")
    s.add_argument("--elevate", action="store_true", help="start it as administrator via a UAC prompt")
    s.set_defaults(func=cmd_player)
    sub.add_parser("stop", help="stop the player").set_defaults(func=cmd_stop)

    s = sub.add_parser("calibrate", help="find her on screen and save her name tag template")
    s.add_argument("--tag", help="name tag centre X,Y in client pixels, if it isn't found automatically")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("snap", help="save a screenshot of the game to gamelab/lab/snap.png")
    s.add_argument("--scale", type=int, default=2)
    s.set_defaults(func=cmd_snap)

    s = sub.add_parser("lock", help="measure how long an attack blocks the next jump")
    s.add_argument("action", choices=["attack", "lucky_seven"])
    s.add_argument("delays", type=int, nargs="+", help="ms after the attack key to press jump")
    s.set_defaults(func=cmd_lock)

    s = sub.add_parser("combo", help="play a combo as the helper would and check jumps and throws")
    s.add_argument("name", choices=[c.name for c in COMBOS])
    s.add_argument("--runs", type=int, default=1)
    s.add_argument("--chain", type=int, default=1, help="back-to-back copies per run, like a queued press")
    s.add_argument("--regular-attack", action="store_true", help="throw with the plain attack key (no MP)")
    s.set_defaults(func=cmd_combo)

    s = sub.add_parser("sheet", help="contact sheet of a recording")
    s.add_argument("recording")
    s.add_argument("--start", type=float, default=-1e9)
    s.add_argument("--end", type=float, default=1e9)
    s.add_argument("--every", type=int, default=3)
    s.add_argument("--cols", type=int, default=8)
    s.add_argument("--around-her", action="store_true", help="zoom in on her instead of the whole crop")
    s.set_defaults(func=cmd_sheet)

    s = sub.add_parser("check", help="key send lag and game frame pacing of a recording")
    s.add_argument("recording")
    s.set_defaults(func=cmd_check)

    sub.add_parser("list", help="list saved recordings").set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (client.PlayerError, RuntimeError) as e:
        print(f"error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
