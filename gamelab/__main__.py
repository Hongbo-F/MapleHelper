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
from .timeline import action_events, chain, her_setup
from .track import airborne, track


def _game_title() -> str:
    return her_setup()[0].game_window_title


def cmd_player(args) -> int:
    client.check_alarm()
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


def cmd_clear_alarm(args) -> int:
    from .alarm import clear_alarm
    info = clear_alarm()
    if info is None:
        print("no Lie Detector alarm is set")
        return 0
    print(f"cleared the Lie Detector alarm from {info.get('time', '?')} (screenshot: {info.get('screenshot', 'none')})")
    print("gamelab can be started again. The game flagged automated play once; keep sessions short and supervised.")
    return 0


def cmd_spot(args) -> int:
    """Check one image, or the game right now, for the Lie Detector (read-only, no keys)."""
    from .alarm import LieDetectorSpotter
    if args.image:
        frame = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        import mss
        hwnd = window.find_game(_game_title())
        with mss.MSS() as sct:
            frame = np.asarray(sct.grab(window.client_rect(hwnd)))[:, :, 2::-1]
    spotter = LieDetectorSpotter()
    score = spotter.score(frame)
    print(f"Lie Detector score {score:.3f} (threshold {spotter.threshold}): "
          f"{'LIE DETECTOR ON SCREEN' if score >= spotter.threshold else 'not on screen'}")
    return 0


def cmd_calibrate(args) -> int:
    window.bring_to_front(_game_title())
    tag = tuple(int(v) for v in args.tag.split(",")) if args.tag else None
    cal = calibration.calibrate(tag, fresh=args.fresh)
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


def parse_steps(steps: list[str], tap_ms: int = 40) -> tuple[list, int]:
    """Explore-step mini language -> (plan events, end ms).

    KEY       tap the key          KEY*MS   hold it MS ms        KEY+ / KEY-   press / release
    MS        wait MS ms           @X,Y     move the mouse to client pixel X,Y (no click)
    Steps run one after another; KEY+ / KEY- let a key stay held across later steps (e.g. right+ alt right-).
    """
    events, t = [], 0
    for step in steps:
        if step.isdigit():
            t += int(step)
        elif step.startswith("@"):
            x, y = (int(v) for v in step[1:].split(","))
            events.append([t, "mouse", x, y])
        elif step.endswith(("+", "-")) and len(step) > 1:
            events.append([t, step[:-1], "down" if step.endswith("+") else "up"])
        else:
            name, _, hold = step.partition("*")
            ms = int(hold) if hold else tap_ms
            events += [[t, name, "down"], [t + ms, name, "up"]]
            t += ms
    return events, t


def cmd_act(args) -> int:
    """Run a few explore steps, then save a screenshot (and optionally a recording around her)."""
    window.bring_to_front(_game_title())
    events, end = parse_steps(args.steps)
    duration = end + args.settle
    if args.record:
        cal = calibration.load()
        rec = client.record(events, name=args.name, duration_ms=duration, crop=cal.crop, scale=cal.scale, fps=30)
        path = LAB / f"{args.name}_sheet.png"
        client.contact_sheet(rec.frames, rec.times, path, cols=8, every=args.every)
        print(path)
    if args.record:  # the steps already ran: just take the screenshot
        frame = client.submit(fps=10, duration_ms=100, scale=args.scale, lead_ms=0, name="act")["frames"][-1]
    else:  # a few full-size frames while the steps run; only the last is kept
        fps = max(2, min(10, int(9000 / (duration + 100))))
        frame = client.submit(events, fps=fps, duration_ms=max(duration, 200), scale=args.scale, lead_ms=0,
                              name="act")["frames"][-1]
    path = LAB / f"{args.name}.png"
    Image.fromarray(frame).save(path)
    print(path)
    return 0


def cmd_do(args) -> int:
    """Run one maplehelper.actions move in the game, recording it, then screenshot."""
    from maplehelper.actions import ACTIONS
    fn, converters = ACTIONS[args.action]
    if len(args.args) > len(converters):
        print(f"{args.action} takes at most {len(converters)} argument(s)")
        return 1
    values = [conv(v) for conv, v in zip(converters, args.args)]
    cfg, keys, timing = her_setup()
    events, end = action_events(fn, values, args.facing, keys, timing)
    if events and events[-1][0] > 20_000:
        print(f"{args.action} would take {events[-1][0] / 1000:.0f} s; gamelab plans are capped at 20 s")
        return 1
    window.bring_to_front(cfg.game_window_title)
    cal = calibration.load()
    rec = client.record(events, name=args.name, duration_ms=end + 400, crop=cal.crop, scale=cal.scale, fps=30)
    client.contact_sheet(rec.frames, rec.times, LAB / f"{args.name}_sheet.png", cols=8, every=args.every)
    frame = client.submit(fps=10, duration_ms=100, scale=2, lead_ms=0, name="do")["frames"][-1]
    Image.fromarray(frame).save(LAB / f"{args.name}.png")
    print(f"{args.action}{tuple(values)}: {len(events)} key events, {end} ms; "
          f"saved {LAB / (args.name + '.png')} and recording '{args.name}'")
    return 0


def _annotate(sight, path) -> None:
    from PIL import ImageDraw
    img = Image.fromarray(sight.frame)
    d = ImageDraw.Draw(img)
    if sight.her:
        x, y = sight.her
        d.rectangle([x - 20, y - 40, x + 20, y + 6], outline="cyan", width=2)
        d.line([0, sight.floor_y, img.width, sight.floor_y], fill="cyan")
        if sight.ground:
            d.line([sight.ground[0], sight.floor_y + 3, sight.ground[1], sight.floor_y + 3], fill="lime", width=3)
    for m in sight.mobs:
        on = m in sight.on_floor()
        d.rectangle([m.x - m.w // 2, m.y - m.h // 2, m.x + m.w // 2, m.y + m.h // 2],
                    outline="red" if on else "orange", width=2)
        d.text((m.x - m.w // 2, m.y - m.h // 2 - 11), f"{m.kind} {m.score:.2f}", fill="red" if on else "orange")
    d.text((8, img.height - 70), f"HP {sight.hp:.0%}  MP {sight.mp:.0%}", fill="white")
    img.save(path)


def cmd_hunt(args) -> int:
    from .hunt import Eyes, Hunter
    cfg, keys, timing = her_setup()
    eyes = Eyes(cfg.game_window_title, calibration.load().template)
    if args.mode == "look":
        sight = eyes.look()
        her = f"at {sight.her}" if sight.her else "NOT FOUND"
        print(f"her: {her}; HP {sight.hp:.0%}, MP {sight.mp:.0%}; room to walk: "
              f"{sight.room('left'):.0f} px left, {sight.room('right'):.0f} px right")
        for m in sight.mobs:
            where = "on her floor" if m in sight.on_floor() else "elsewhere"
            dx = f"{sight.game_dx(m.x):+.0f} px" if sight.her else "?"
            print(f"  {m.kind} {dx} ({where}, score {m.score})")
        _annotate(sight, LAB / "hunt_look.png")
        print(LAB / "hunt_look.png")
        return 0
    kinds = set(args.kinds.split(",")) if args.kinds else None
    hunter = Hunter(eyes, keys, timing, low_hp=args.low_hp, potion_hp=args.potion_hp,
                    log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))
    window.bring_to_front(cfg.game_window_title)
    if args.mode == "kill":
        sight = eyes.look()
        targets = sight.on_floor(kinds)
        if sight.her is None or not targets:
            print("no target on her floor" if sight.her else "can't see her")
            return 1
        near = args.near if args.near is not None else 0
        target = min(targets, key=lambda m: abs(sight.game_dx(m.x) - near))
        hunter.report.stop = hunter.kill(target, sight, seconds=args.seconds)
    else:
        hunter.clear_floor(kinds, max_kills=args.kills, seconds=args.seconds, span=args.span)
    final = eyes.look()
    _annotate(final, LAB / "hunt_end.png")
    print(f"done: {hunter.report}; now HP {final.hp:.0%}, MP {final.mp:.0%}")
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


def cmd_track(args) -> int:
    """Her path through a recording: screen position, height above the start, world x, and jumps."""
    from .camera import world_x
    rec = client.load(args.recording)
    template = calibration.load().template
    rows = track(rec.frames, rec.times, template)
    xs = world_x(rec.frames, rows)
    rest_y = rows[0][2]
    print("   t ms   screen x,y   height   world x   match")
    for i in range(0, len(rows), args.every):
        t, x, y, score = rows[i]
        print(f"  {t:6.0f}   {x:4d},{y:4d}   {rest_y - y:5d}   {xs[i]:7.1f}   {score:.2f}")
    print("airborne:", [(round(a), round(b) if b else None) for a, b in airborne(rows)])
    print(f"net world dx {xs[-1] - xs[0]:+.1f} px (recording scale {calibration.load().scale})")
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
    sub.add_parser("clear-alarm", help="after you've answered a Lie Detector: let gamelab run again"
                   ).set_defaults(func=cmd_clear_alarm)
    s = sub.add_parser("spot", help="check the game (or an image) for the Lie Detector; sends no keys")
    s.add_argument("--image", help="check this screenshot instead of the game")
    s.set_defaults(func=cmd_spot)

    s = sub.add_parser("calibrate", help="find her on screen and save her name tag template")
    s.add_argument("--tag", help="name tag centre X,Y in client pixels, if it isn't found automatically")
    s.add_argument("--fresh", action="store_true", help="rebuild the name tag template (stand clear of ropes)")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("snap", help="save a screenshot of the game to gamelab/lab/snap.png")
    s.add_argument("--scale", type=int, default=2)
    s.set_defaults(func=cmd_snap)

    s = sub.add_parser("act", help="run explore steps (keys, waits, mouse moves), then screenshot",
                       description=parse_steps.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("steps", nargs="+", help="e.g.  right*800  alt  500  up*1500  @960,540  \\\\")
    s.add_argument("--name", default="act", help="file name for the screenshot / recording")
    s.add_argument("--settle", type=int, default=400, help="ms to wait after the last step before the screenshot")
    s.add_argument("--scale", type=int, default=2, help="screenshot downscale")
    s.add_argument("--record", action="store_true", help="also record her at 30 fps and save a contact sheet")
    s.add_argument("--every", type=int, default=3, help="contact sheet: keep every Nth frame")
    s.set_defaults(func=cmd_act)

    from maplehelper.actions import ACTIONS
    s = sub.add_parser("do", help="run a maplehelper.actions move in the game (walk, climb_rope, pick_up, ...)")
    s.add_argument("action", choices=sorted(ACTIONS))
    s.add_argument("args", nargs="*", help="e.g.  walk right 200   climb_rope 300   pick_up 2   rest 100")
    s.add_argument("--facing", choices=[LEFT, RIGHT], default=RIGHT, help="which way she faces now (stand_up)")
    s.add_argument("--name", default="do", help="recording / screenshot name")
    s.add_argument("--every", type=int, default=4)
    s.set_defaults(func=cmd_do)

    s = sub.add_parser("hunt", help="hunting objectives that run on their own (see gamelab/hunt.py)")
    s.add_argument("mode", choices=["look", "kill", "clear"],
                   help="look: what she sees now; kill: one monster; clear: her floor, nearest first")
    s.add_argument("--kinds", help="only these monsters, e.g. slime,pig")
    s.add_argument("--near", type=float, help="kill: the target closest to this many px from her (+ = right)")
    s.add_argument("--kills", type=int, default=6, help="clear: stop after this many kills")
    s.add_argument("--seconds", type=float, default=120, help="stop after this long")
    s.add_argument("--span", type=float, default=900, help="clear: stay within this many px of the start")
    s.add_argument("--potion-hp", type=float, default=0.5, help="drink an HP potion below this HP fraction")
    s.add_argument("--low-hp", type=float, default=0.35, help="stop and hand back below this HP fraction")
    s.set_defaults(func=cmd_hunt)

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

    s = sub.add_parser("track", help="print her path through a recording (position, height, world x)")
    s.add_argument("recording")
    s.add_argument("--every", type=int, default=3)
    s.set_defaults(func=cmd_track)

    s = sub.add_parser("check", help="key send lag and game frame pacing of a recording")
    s.add_argument("recording")
    s.set_defaults(func=cmd_check)

    sub.add_parser("list", help="list saved recordings").set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    # Real pixels for window positions and screen captures, whatever the display scaling is set to.
    from maplehelper.win32 import enable_dpi_awareness
    enable_dpi_awareness()
    try:
        if args.cmd in GAME_COMMANDS:
            client.check_alarm()
        return args.func(args)
    except client.LieDetectorAlarm as e:
        print(f"LIE DETECTOR: {e}. gamelab stopped all input and the player shut down.\n"
              "Answer it in the game yourself. Afterwards, run `python -m gamelab clear-alarm`.")
        return LIE_DETECTOR_EXIT
    except (client.PlayerError, RuntimeError) as e:
        print(f"error: {e}")
        return 1


# Commands that bring the game to the front or send it input; refused while the alarm is set.
GAME_COMMANDS = {"player", "calibrate", "snap", "act", "do", "lock", "combo", "hunt"}
LIE_DETECTOR_EXIT = 3


if __name__ == "__main__":
    raise SystemExit(main())
