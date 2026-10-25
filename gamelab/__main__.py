"""gamelab's command line (`python -m gamelab -h`): the key player, calibration, exploring and measuring
moves with recordings, hunting, navigation and tours in the game, the mini-map, and the simulator.

The top of the stack: it may import every layer. Only what many commands share (analysis, control and
the shared modules) is imported up here; the other layers are imported by the commands that use them.
"""

import argparse
import ctypes
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from maplehelper.combos import COMBOS
from maplehelper.engine import LEFT, RIGHT

from . import paths, telemetry, window
from .analysis import analyze, calibration, recordings
from .analysis.track import airborne, track
from .control import client
from .control.timeline import action_events, chain, her_setup
from .paths import LAB, PLAYER_LOG, RECORDINGS, REPO, SHOTS, STOP
from .screen import STRETCHED, geometry

DEFAULT_MAP = 10000012  # Victoria Road: Thicket Around the Beach II
FRONT_WAIT_S = 3.0  # s a hunting command waits for the game to come to the front before it gives up (_to_game)
MINIMIZED_POLL_S = 0.5  # s between looks at whether the minimized game was restored (`watch` waits: _game_geometry)
MINIMAP_TRIES = 3       # times the map's minimap picture is tried before a command that acts stops (_minimap_picture),
MINIMAP_RETRY_S = 0.5   # ... this long apart (in case what kept it from loading passes)
# `watch`: how often it publishes what it sees (telemetry), and says how it's going on the console. Scenes 10
# times a second and tracks twice a second: on the saved frames a scene is 393 bytes (with what each look searched,
# for analysis.respawn, and the other players; 279 before those) and the tracks 848, so about 20 MB an hour, 490 MB
# a day: the 2 GB telemetry keeps (telemetry.KEEP_BYTES) hold 4 days of it (5 before).
WATCH_SCENE_S = 0.1
WATCH_TRACKS_S = 0.5
WATCH_STATUS_S = 10.0
WATCH_LIE_DETECTOR_S = 0.5  # --lie-detector: s between checks of the screen for it (the player checks 4 a second)
WATCH_EVENTS = ("died", "exp_gain")  # tracker events published as telemetry 'event' records (and hits, new ones)
# Commands that bring the game to the front or send it input; refused while the alarm is set. So is any
# subcommand whose parser sets `game` (`probe run`; `probe list` and `probe score` send nothing).
GAME_COMMANDS = {"player", "calibrate", "snap", "act", "do", "lock", "combo", "hunt", "nav", "tour"}
LIE_DETECTOR_EXIT = 3   # exit code after the Lie Detector stopped a command

SW_SHOWMINNOACTIVE = 7  # ShellExecute: start the player minimized, not activated
SHELL_ERRORS = {5: "the UAC prompt was declined", 2: "python not found", 3: "folder not found"}

TAP_MS = 40             # a key tap: down, and up this much later
SETTLE_S = 0.6          # rest between recorded runs (lock, combo)
MOVES_BACK = {"jump_back_shot", "hop_back"}  # combos whose jumps must carry her away
FACE_MS = 600  # each run first taps the arrow she should face, like her own last arrow press
SHOT_BAND = 30          # `hunt shots` groups throws by the target's height in bands this tall (map units)
# `sheet --around-her`: a box around her name tag (x, y from its top-left corner, width, height; she
# stands above it), shown this many times larger.
AROUND_HER = (-40, -60, 120, 80)
AROUND_HER_ZOOM = 3


def _game_title() -> str:
    return her_setup()[0].game_window_title


def _game_geometry(title: str, restore: bool = False):
    """The game window's geometry (screen.Geometry, from its client size). Minimized, its client area isn't the
    game's (GetClientRect gave 432x54 on 2026-10-24), so it's read once the window is back: with `restore` (the
    commands that bring the game to the front anyway: their first look did, before its size was read here) the
    window is restored and brought forward first (window.bring_to_front); without (`watch`, which never brings it
    forward), this waits until whoever is at the PC restores it. Otherwise it reads the window only."""
    hwnd = window.find_game(title)
    if window.minimized(hwnd):
        if restore:
            window.bring_to_front(title, wait=FRONT_WAIT_S)
            if window.minimized(hwnd):
                raise RuntimeError("the game is minimized and didn't come back: restore it, then run this again")
        else:
            print("the game is minimized: waiting for it to be restored (nothing is read; Ctrl+C stops)", flush=True)
            while window.minimized(hwnd):
                time.sleep(MINIMIZED_POLL_S)
    area = window.client_rect(hwnd)
    return geometry(area["width"], area["height"])


# --- the player and the Lie Detector ---------------------------------------------------------------------

def cmd_player(args) -> int:
    client.check_alarm()
    if not args.elevate:
        from .control.player import serve
        serve()
        return 0
    # ShellExecute "runas" shows the UAC prompt and starts `python -m gamelab player` elevated.
    rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, "-X utf8 -m gamelab player",
                                             str(REPO), SW_SHOWMINNOACTIVE)
    if rc <= 32:  # ShellExecute's error codes
        reason = SHELL_ERRORS.get(rc, "see ShellExecute docs")
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
    from .control import lie_recorder
    from .control.alarm import alarm_info, clear_alarm
    folder = lie_recorder.going()
    if folder is not None and alarm_info() is not None:
        # Cleared now, a running player would stop a minute after the Lie Detector's last sight, the person maybe
        # still at it (the spotter can't see the figure tracking: lie_recorder.END_S); and the commands would run.
        print(f"the Lie Detector's recording is still going ({folder}): it ends once the game has shown no Lie "
              f"Detector for {lie_recorder.END_S:g} s (with no window over it: a window over the game, or the game "
              "minimized, doesn't count). The alarm stays up: run `clear-alarm` again then.")
        return LIE_DETECTOR_EXIT
    info = clear_alarm()
    if info is None:
        print("no Lie Detector alarm is set")
    else:
        print(f"cleared the Lie Detector alarm from {info.get('time', '?')} "
              f"(screenshot: {info.get('screenshot', 'none')})")
    if folder is not None:
        print(f"the Lie Detector recording is still going: {folder} (until it has been off the screen for "
              f"{lie_recorder.END_S:g} s)")
    elif folders := lie_recorder.recordings():
        summary = lie_recorder.summary_of(folders[-1])
        print(f"the last recording: {folders[-1]} " + (
            f"({summary['outcome']}: {summary['frames']} frames over {summary['duration_s']:.0f} s, ended: "
            f"{summary['end']})" if summary else "(unfinished: the player stopped while it was written)"))
    if info is not None:
        print(f"a running player stops now (or, if the Lie Detector is still up, once it has been off the screen "
              f"for {lie_recorder.END_S:g} s): gamelab's commands need it started again, `python -m gamelab player "
              "--elevate`. The game flagged automated play once; keep sessions short and supervised.")
    return 0


def cmd_spot(args) -> int:
    """Check one image, or the game right now, for the Lie Detector (read-only, no keys)."""
    from .perception.lie_detector import LieDetectorSpotter
    if args.image:
        frame = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        import mss
        hwnd = window.find_game(_game_title())
        with mss.MSS() as sct:
            frame = np.asarray(sct.grab(window.client_rect(hwnd)))[:, :, 2::-1]
    spotter = LieDetectorSpotter()
    scores = spotter.scores(frame)
    seen = max(scores.values()) >= 1
    print(f"{'LIE DETECTOR ON SCREEN' if seen else 'not on screen'} (match / threshold, 1 = seen: "
          + ", ".join(f"{name} {v:.2f}" for name, v in scores.items()) + ")")
    return 0


# --- calibrating, exploring, recording -------------------------------------------------------------------

def cmd_calibrate(args) -> int:
    window.bring_to_front(_game_title())
    tag = tuple(int(v) for v in args.tag.split(",")) if args.tag else None
    cal = calibration.calibrate(tag, fresh=args.fresh)
    print(f"recording crop {cal.crop} (client px), name tag template {cal.template.shape[1]}x{cal.template.shape[0]}")
    print(f"check {calibration.pictures(cal.geo)[1]}: her name tag should be near the middle, a bit below centre")
    return 0


def _shot(name: str) -> Path:
    """Where a command saves a picture: gamelab/lab/shots (the lab's own screenshots are test data: `snap` and
    `hunt look` once wrote over two of them)."""
    shots = LAB / SHOTS.name  # LAB as it is when called (tests point it elsewhere)
    shots.mkdir(parents=True, exist_ok=True)
    return shots / name


def cmd_window(args) -> int:
    """Resize the game's client area (through the player: a normal program can't), then save a full-size screenshot
    of it. Sends no keys."""
    width, height = (int(v) for v in args.size.lower().split("x"))
    result = client.submit(fps=10, duration_ms=200, scale=1, lead_ms=0, name="window", window_client=(width, height))
    path = _shot(f"window_{width}x{height}.png")
    Image.fromarray(result["frames"][-1]).save(path)
    print(f"client area now {result['client']['width']}x{result['client']['height']}: {path}")
    return 0


def cmd_snap(args) -> int:
    window.bring_to_front(_game_title())
    frame = recordings.screenshot("snap", scale=args.scale)
    path = _shot("snap.png")
    Image.fromarray(frame).save(path)
    print(path)
    return 0


def parse_steps(steps: list[str], tap_ms: int = TAP_MS) -> tuple[list, int]:
    """Explore-step mini language -> (plan events, end ms).

    KEY       tap the key          KEY*MS   hold it MS ms        KEY+ / KEY-   press / release
    MS        wait MS ms           @X,Y     move the mouse to client pixel X,Y
    click@X,Y / rclick@X,Y         move the mouse there and click (left / right button)
    Steps run one after another; KEY+ / KEY- let a key stay held across later steps (e.g. right+ alt right-).
    """
    events, t = [], 0
    for step in steps:
        if step.isdigit():
            t += int(step)
        elif step.startswith(("click@", "rclick@")):
            button, _, at = step.partition("@")
            x, y = (int(v) for v in at.split(","))
            name = "lbutton" if button == "click" else "rbutton"
            events += [[t, "mouse", x, y], [t, name, "down"], [t + tap_ms, name, "up"]]
            t += tap_ms
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


def _tap(key: str, at_ms: int) -> list[list]:
    """Plan events that tap `key` at `at_ms`."""
    return [[at_ms, key, "down"], [at_ms + TAP_MS, key, "up"]]


def _recording_setup() -> tuple[calibration.Calibration, list[int]]:
    """The calibration for the game window at the size it is now, and the crop to record her in: around her name
    tag as the game shows it now, or the stored one if the tag isn't found (calibration.crop_on_her, on a short
    capture through the player, no keys, which also says the window's size); what it found is said."""
    full = client.submit(fps=10, duration_ms=100, scale=1, lead_ms=0, name="find_her")
    w, h = full["client"]["width"], full["client"]["height"]
    cal = _calibration(geometry(w, h))
    crop, note = calibration.crop_on_her(cal, full["frames"][-1], w, h)
    if note:
        print(note)
    return cal, crop


def _calibration(geo) -> calibration.Calibration:
    """The saved calibration for a game window of geometry `geo` (calibration.load)."""
    return calibration.load() if geo == STRETCHED else calibration.load(geo)


def cmd_act(args) -> int:
    """Run a few explore steps, then save a screenshot (and optionally a recording around her: where her name tag
    is when it starts, not where calibrate found her)."""
    window.bring_to_front(_game_title())
    events, end = parse_steps(args.steps)
    duration = end + args.settle
    if args.record:
        cal, crop = _recording_setup()
        rec = recordings.record(events, name=args.name, duration_ms=duration, crop=crop, scale=cal.scale, fps=30)
        path = _shot(f"{args.name}_sheet.png")
        recordings.contact_sheet(rec.frames, rec.times, path, cols=8, every=args.every)
        print(path)
        frame = recordings.screenshot("act", scale=args.scale, duration_ms=100)  # the steps already ran
    else:  # a few full-size frames while the steps run (~9 over the run, 2-10 fps); only the last is kept
        fps = max(2, min(10, int(9000 / (duration + 100))))
        frame = client.submit(events, fps=fps, duration_ms=max(duration, 200), scale=args.scale, lead_ms=0,
                              name="act")["frames"][-1]
    path = _shot(f"{args.name}.png")
    Image.fromarray(frame).save(path)
    print(path)
    return 0


def cmd_do(args) -> int:
    """Run one maplehelper.actions move in the game, recording it, then screenshot."""
    from maplehelper.actions import ACTIONS

    from .control.player import MAX_PLAN_MS
    fn, converters = ACTIONS[args.action]
    if len(args.args) > len(converters):
        print(f"{args.action} takes at most {len(converters)} argument(s)")
        return 1
    values = [conv(v) for conv, v in zip(converters, args.args)]
    cfg, keys, timing = her_setup()
    events, end = action_events(fn, values, args.facing, keys, timing)
    if events and events[-1][0] > MAX_PLAN_MS:
        print(f"{args.action} would take {events[-1][0] / 1000:.0f} s; "
              f"gamelab plans are capped at {MAX_PLAN_MS // 1000} s")
        return 1
    window.bring_to_front(cfg.game_window_title)
    cal, crop = _recording_setup()
    rec = recordings.record(events, name=args.name, duration_ms=end + 400, crop=crop, scale=cal.scale, fps=30)
    recordings.contact_sheet(rec.frames, rec.times, _shot(f"{args.name}_sheet.png"), cols=8, every=args.every)
    path = _shot(f"{args.name}.png")
    Image.fromarray(recordings.screenshot("do", scale=2, duration_ms=100)).save(path)
    print(f"{args.action}{tuple(values)}: {len(events)} key events, {end} ms; "
          f"saved {path} and recording '{args.name}'")
    return 0


# --- hunting, the map, navigation, tours -----------------------------------------------------------------

def _box(thing) -> list:
    """[left, top, right, bottom] of something seen on screen, from its centre (x, y) and size (w, h)."""
    return [thing.x - thing.w // 2, thing.y - thing.h // 2, thing.x + thing.w // 2, thing.y + thing.h // 2]


def _annotate(scene, path) -> None:
    """Save a look's frame with what it found drawn on: her name tag and floor, the monsters (red on her
    platform, orange elsewhere), the drops, and her HP and MP."""
    from PIL import ImageDraw
    img = Image.fromarray(scene.frame)
    d = ImageDraw.Draw(img)
    if scene.tag:
        x, y = scene.tag
        colour = "cyan" if scene.tag_seen else "magenta"  # magenta: placed from the minimap dot
        d.rectangle([x - 20, y - 40, x + 20, y + 6], outline=colour, width=2)
        d.line([0, scene.floor_y, img.width, scene.floor_y], fill=colour)
    for monster in scene.monsters:
        mine = scene.platform is not None and monster.platform == scene.platform.id
        colour = "red" if mine else "orange"
        mob = monster.mob
        box = _box(mob)
        d.rectangle(box, outline=colour, width=2)
        d.text((box[0], box[1] - 11), f"{mob.kind} p{monster.platform}", fill=colour)
        x, y = mob.foot_x, mob.feet_y  # where it stands, and an arrow the way it faces
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=colour)
        if mob.facing:
            d.line([x, y - 4, x + 10 * mob.facing, y - 4], fill=colour, width=2)
            d.polygon([(x + 14 * mob.facing, y - 4), (x + 9 * mob.facing, y - 8), (x + 9 * mob.facing, y)], fill=colour)
    for loot in scene.loot:
        box = _box(loot.item)
        d.rectangle(box, outline="yellow", width=2)
        d.text((box[0], box[3] + 2), f"{loot.item.kind} p{loot.platform}", fill="yellow")
    d.text((8, img.height - 70), f"HP {scene.hp:.0%}  MP {scene.mp:.0%}", fill="white")
    img.save(path)


def _eyes(model, title: str):
    """Eyes on the game, with monster templates for that map (screenshots + the game's sprites), for its window at
    the size it is now (its calibration at that size), and the map's minimap picture (_minimap_picture). Says where
    the window is, at what DPI awareness, and what minimap picture was loaded: what a run that went wrong is told
    by (2026-10-24 16:46:50)."""
    from .perception.eyes import Eyes
    from .perception.sprites import detector_for
    geo = _game_geometry(title, restore=True)  # (a minimized game is brought back, as the first look would)
    area = window.client_rect(window.find_game(title))
    print(f"the game's client: {area['width']}x{area['height']} at ({area['left']}, {area['top']}) (desktop px); "
          f"DPI awareness {window.dpi_awareness()}", flush=True)
    return Eyes(title, _calibration(geo).template, detector_for(model, geo), geo=geo,
                minimap_picture=_minimap_picture(model))


def _minimap_picture(model):
    """The map's minimap picture (mapdata.load_minimap_picture), tried MINIMAP_TRIES times, MINIMAP_RETRY_S apart; None
    for a map without one. Raises RuntimeError, saying why, if the map has one and it won't load: without it her dot
    is read with the first map's origin, which on Tree Dungeon put her 640 units off the map for a whole hunt (a
    process in which it was None, 2026-10-24 16:46:50; why isn't known: mapdata.minimap_picture said nothing)."""
    from .localization import mapdata
    why = None
    for attempt in range(MINIMAP_TRIES):
        try:
            picture = mapdata.load_minimap_picture(model.id)
        except (KeyError, OSError, ValueError) as e:
            why = e
            if attempt + 1 < MINIMAP_TRIES:
                time.sleep(MINIMAP_RETRY_S)
            continue
        print(f"the minimap picture of map {model.id}: {picture.shape[1]}x{picture.shape[0]} game px", flush=True)
        return picture
    if (getattr(model, "minimap", None) or {}).get("width"):
        raise RuntimeError(f"map {model.id}'s minimap picture didn't load ({why!r}): without it her minimap dot is read "
                           f"with the first map's origin (on Tree Dungeon 640 units off the map)")
    print(f"map {model.id} has no minimap picture: her dot is read with the first map's origin", flush=True)
    return None


_FEEDS: list = []  # continuous feeds started by this command (--feed), closed when it ends


def _hunter(args, **kw):
    """The hunter on map `args.map`, logging to the console (sim.scenarios builds its stack the same way); with
    `--feed`, her looks and glances from a continuous feed (localization.feed: the game looked at every frame
    in a process of its own)."""
    from .behavior.hunter import Hunter
    from .localization import mapdata
    cfg, keys, _ = her_setup()
    model = mapdata.load(args.map)
    eyes = _eyes(model, cfg.game_window_title)
    feed = _feed(model, cfg.game_window_title, eyes.geo) if getattr(args, "feed", False) else None
    return Hunter(eyes, keys, model, log=_log, feed=feed, sweep=bool(getattr(args, "sweep", None)), **kw)


def cmd_pet(args) -> int:
    """Her pet kept fed (behavior.pet): its fullness followed in time and a food given when it's low, read once off her
    Character Info to start from (or now, with --read); with --every, followed every that many minutes until stopped,
    counting only time the game window is there. Keys: pet food; for a read Esc and a double-click on her. With the Lie
    Detector alarm up it goes on: the food by the fullness followed (the player takes it then), no read until the alarm
    is cleared; a food the player refuses or doesn't answer (it isn't running) is tried again at the next follow."""
    hunter = _hunter(args)
    _record("pet", args.map)
    title = _game_title()
    if args.read and client.alarm_up():
        print("pet fullness: not read with the Lie Detector alarm up (the read double-clicks her)")
    elif args.read:
        window.bring_to_front(title)
        print(f"pet fullness: {hunter.pet.check()}")
    if args.read and not args.every:
        return 0
    while True:
        try:
            present = not window.minimized(window.find_game(title))
        except RuntimeError:  # no game window: the pet isn't out
            present = False
        try:
            level = hunter.pet.tend(present=present)
        except client.PlayerError as e:  # the alarm's limits, the player stopped: the food goes again next time
            _log(f"pet: no food given ({type(e).__name__}: {e}); tried again at the next follow")
        else:
            print(f"pet fullness: {'not known' if level is None else f'about {level:.0f}'}"
                  + ("" if present else " (the game isn't there: time not counted)"))
        if not args.every:
            return 0
        time.sleep(args.every * 60)


def _feed(model, title: str, geo=STRETCHED):
    """A continuous feed of the game for this command (its child process; closed when the command ends), for its
    window at geometry `geo` (the command's eyes'), or None if it can't start (the command then looks for itself)."""
    import functools

    from .attack import reach
    from .localization import feed
    try:
        ring = {} if geo == STRETCHED else {"shape": feed.frame_shape(geo)}  # (FeedClient's own default otherwise)
        client = feed.FeedClient(functools.partial(feed.screen_world, model.id, title, _calibration(geo).template,
                                                   (reach.LOOK_UP, reach.LOOK_DOWN), geo=geo), **ring)
    except Exception as e:  # it couldn't even be started: the command goes on without
        _log(f"feed: couldn't start it ({type(e).__name__}: {e}); looking without it")
        return None
    if not client.alive:
        _log(f"feed: {client.why}; looking without it")
        client.close()
        return None
    _FEEDS.append(client)
    _log(f"feed: the game looked at {client.hz:.0f} times a second in process {client.hello['pid']} "
         f"(below normal priority: {client.hello['priority'] == feed.BELOW_NORMAL})")
    return client


def _close_feeds() -> None:
    """Stop the command's feeds, and say what each did. Its child sends its counts (`bye`) only once asked to
    stop, so they're read after close: read before it, on 2026-10-21 (`tour --feed`), every count was '?'."""
    while _FEEDS:
        client = _FEEDS.pop()
        down = "" if client.alive else client.why  # it went down before the command ended
        client.close()
        bye = client.bye or {}
        print(f"feed: {bye.get('frames', '?')} frames, {bye.get('placed', '?')} placing her, "
              f"{bye.get('band', '?')} band and {bye.get('wide', '?')} whole-screen searches; "
              f"{client.served} looks and glances from it, {client.refused} taken without it"
              + (f"; {down}" if down else ""))


def _to_game() -> bool:
    """Bring the game to the front before a hunting command's first look and plan, and see that it came: the
    player refuses every plan while another window is in front, and a look captures whatever is (on 2026-10-21
    at 01:01:30 a tour's first looks saw another window for 2.5 s and its first plan was refused, "game not in
    front"). If Windows won't bring it forward (window.bring_to_front, FRONT_WAIT_S), the command says so and
    doesn't start."""
    title = _game_title()
    if window.bring_to_front(title, wait=FRONT_WAIT_S):
        return True
    print(f"the game didn't come to the front ({window.foreground_title()!r} is in front): click on it, then run "
          "this again")
    return False


def _log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)
    telemetry.emit("log", text=message)


def _record(command: str, map_id: int) -> None:
    """Record this run's telemetry (lab/telemetry), for the mini-map to follow live or replay later."""
    path = telemetry.record_to(command)
    telemetry.emit("session", map=map_id, command=command, mode="live")
    _log(f"telemetry: {path} (watch it with `python -m gamelab minimap`: a window, and a page for a phone)")


def cmd_hunt(args) -> int:
    if args.mode == "shots":
        for path in (paths.OLD_SHOT_LOG, paths.SHOT_LOG):
            _shot_summary(path)
        return 0
    hunter = _hunter(args, low_hp=args.low_hp, potion_hp=args.potion_hp, skill=args.skill)
    if args.mode == "look":
        return _hunt_looks(hunter, args.every, args.seconds) if args.every else _hunt_look(hunter)
    kinds = set(args.kinds.split(",")) if args.kinds else None
    _record(f"hunt-{args.mode}", hunter.model.id)
    if not _to_game():
        return 1
    from .behavior.retreat import on_alarm
    with on_alarm(hunter):  # the Lie Detector seen: she retreats to a safe place, then the command stops
        if args.mode == "kill":
            s = hunter.look()
            targets = [t for t in s.monsters if kinds is None or t.kind in kinds]
            if s.where is None or not targets:
                print("no target in sight" if s.where else "can't find her")
                return 1
            near = args.near if args.near is not None else 0
            target = min(targets, key=lambda t: abs(t.x - s.where.x - near))
            hunter.report.stop = hunter.kill(target, seconds=args.seconds)
        else:
            ground = {int(v) for v in args.ground.split(",")} if args.ground else None
            hunter.clear(kinds, ground=ground, max_kills=args.kills, seconds=args.seconds)
            hunter.looter.mop_up()  # drops still on her platform: kills the fight couldn't place, or left before
        final = hunter.look(loot=True)
    _annotate(final, _shot("hunt_end.png"))
    left = ", ".join(f"{d.item.kind} at {d.x - final.where.x:+.0f}" for d in final.loot) or "none"
    print(f"done: {hunter.report}; now HP {final.hp:.0%}, MP {final.mp:.0%}; drops left on her platform: {left}"
          + ("; her inventory is full" if hunter.looter.inventory_full else ""))
    return 0


def _hunt_looks(hunter, every: float, seconds: float) -> int:
    """`hunt look --every S`: a look every S seconds for `seconds`, her place in a line each (no keys; a look brings the
    game to the front, as any look does): what looks make of the game window moved, covered, minimized or resized while
    it runs. One that fails its checks says why, and after senses.LOST_S of failing it stands still until one passes
    (Senses.regain), as a hunt does (README: Capture checks)."""
    from .perception.framecheck import LookLost
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            s = hunter.look(mobs=False)
        except LookLost as lost:
            if hunter.senses.regain(lost, end) is None:
                return 1
            continue
        p = s.platform
        _log(f"her: map ({s.where.x:.0f}, {s.where.y:.0f}) on platform {p.id if p else None} ({s.how})"
             if s.where is not None else "her minimap dot isn't seen")
        time.sleep(every)
    return 0


def _hunt_look(hunter) -> int:
    """`hunt look`: what she sees now, printed and drawn on the frame (no keys)."""
    from .attack import reach
    from .localization.scene import find_loot
    from .perception.minimap import sizes
    s = hunter.look()
    if s.where is None:
        print("can't find her minimap dot")
        return 1
    how = "name tag" if s.tag_seen else "minimap dot + camera (name tag hidden)"
    p = s.platform
    print(f"her: map ({s.where.x:.0f}, {s.where.y:.0f}) on platform {p.id if p else None} ({how}); "
          f"HP {s.hp:.0%}, MP {s.mp:.0%}; room: {s.room('left', reach.EDGE_MARGIN):.0f} left, "
          f"{s.room('right', reach.EDGE_MARGIN):.0f} right")
    exp = _exp_reading(hunter, s)
    if exp is not None and exp.exp is not None:
        print(f"  EXP {exp.text.strip()}: {exp.exp:,} into level {'/'.join(map(str, exp.levels))} (every glyph within "
              f"{exp.fit:.1f} levels of the game's, every other reading {exp.lead:.0f} or more off)")
    else:
        print(f"  EXP number not read: {exp.why if exp is not None else 'no box'}")
    shift = sizes(s.geo).other_shift
    for m in s.others:  # other players' minimap dots (perception.minimap), placed as hers is
        w = hunter.senses.localizer.where_dot((m.x - shift, m.y - shift))
        print(f"  another player ({m.kind}): map ({w.x:.0f}, {w.y:.0f}) on platform "
              f"{w.platform.id if w.platform else None} (their minimap dot at {m.x:.1f}, {m.y:.1f})")
    for t in s.monsters:
        dx, dy = t.x - s.where.x, t.y - s.where.y
        stop = reach.wall_to(p, s.where.x, s.where.y, t)
        in_reach = ("in reach" if reach.can_hit(p, s.where.x, s.where.y, t, others=s.monsters)
                    else "out of reach" if not reach.can_shoot(dx, dy, kind=t.kind)
                    else "behind a wall" if stop is not None and stop.x1 == stop.x2
                    else "a floor in the way" if stop is not None
                    else "behind one whose star is stopped")
        facing = {1: ", facing right", -1: ", facing left"}.get(t.facing, "")
        frame = f", {t.mob.template} ({t.mob.frame})" if t.mob.frame else f", {t.mob.template}"
        print(f"  {t.kind} on platform {t.platform}: {dx:+.0f}, {dy:+.0f} ({in_reach}, score {t.mob.score}{facing}{frame})")
    s.loot = find_loot(s, hunter.model, hunter.senses.eyes, wide=True)
    for d in s.loot:
        print(f"  drop: {d.item.kind} on platform {d.platform}: {d.x - s.where.x:+.0f}, "
              f"{d.y - s.where.y:+.0f} (score {d.item.score})")
    _annotate(s, _shot("hunt_look.png"))
    print(_shot("hunt_look.png"))
    return 0


def _exp_reading(hunter, s):
    """Her EXP number as `hunt look` reads it (perception.exp_number), off the look's box; a look the feed served has
    none (the feed's frames are half size, the box full size: worldstate.to_scene), so then off a look of its own (a
    capture, no keys). None: no box."""
    from .perception import exp_number
    box = s.exp_box
    if box is None and s.events is not None:  # (a look from the feed: it comes with the feed's events)
        box = hunter.senses.eyes.look().exp_box
    return exp_number.read(box) if box is not None else None


def _shot_summary(path) -> int:
    """Hits and misses from a shot log, by how far below (+) or above (-) her the target was (throws left
    undecided aside: in flight before the one that killed it, or one of two that may have made a bar drop)."""
    import json
    if not path.exists():
        print(f"no shots logged yet ({path})")
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bands: dict[int, list[int]] = {}  # band's dy -> [hits, misses, farthest hit's |dx|]
    for r in rows:
        if r["hit"] is None:
            continue
        b = bands.setdefault(round(r["dy"] / SHOT_BAND) * SHOT_BAND, [0, 0, 0])
        b[0 if r["hit"] else 1] += 1
        if r["hit"]:
            b[2] = max(b[2], abs(r["dx"]))
    undecided = sum(1 for r in rows if r["hit"] is None)
    print(f"{path.name}: {len(rows)} throws logged ({undecided} undecided); by height of the target (map units, "
          f"+ = below her):")
    for dy, (hit, miss, far) in sorted(bands.items()):
        print(f"  dy {dy:+4d}: {hit} hit, {miss} missed; farthest hit {far} units")
    return 0


def cmd_probe(args) -> int:
    """The in-game probes of what a star hits (behavior.probe): `list` them, `run` one in the game (the way
    `hunt kill` runs: the hunter, the player, the Lie Detector alarm), `score` their log offline."""
    from .behavior import probe
    from .localization import mapdata
    if args.action == "list":
        print(probe.listing())
        return 0
    if args.action == "score":
        path = Path(args.file) if args.file else paths.PROBE_LOG
        rows = probe.load(path)
        if not rows:
            print(f"no probe throws logged yet ({path})")
            return 0
        if args.line:  # this command only: it sends nothing, and nothing else runs on it
            from .attack import reach
            reach.LINE = args.line
        print(probe.score(rows, mapdata.load(args.map or rows[0]["map"])))
        return 0
    if args.name not in probe.PROBES:
        print(f"no probe {args.name!r}; they are: {', '.join(probe.PROBES)} (`probe list` says what each settles)")
        return 1
    p = probe.adjust(probe.PROBES[args.name], x=args.x, dx=args.dx, style=args.style, throws=args.throws,
                     walls=None if args.walls is None else args.walls == "on")
    if trouble := probe.problems(mapdata.load(args.map), p):
        print(f"probe {p.name} can't run like that: {'; '.join(trouble)}")
        return 1
    hunter = _hunter(args, low_hp=args.low_hp, potion_hp=args.potion_hp)  # the regular attack: no --skill
    nav = hunter.navigator()
    _record(f"probe-{p.name}", hunter.model.id)
    if not _to_game():
        return 1
    from .behavior.retreat import on_alarm
    with on_alarm(hunter):  # the Lie Detector seen: she retreats to a safe place, then the command stops
        report = probe.run(hunter, nav, p, seconds=args.seconds, crowd=args.crowd)
    print(f"done: {report}. Logged to {paths.PROBE_LOG}: `python -m gamelab probe score` reads it")
    return 0


def _span(text: str) -> tuple[float, float]:
    """`--dx LO-HI`: a range of map units sideways."""
    lo, sep, hi = text.partition("-")
    try:
        span = float(lo), float(hi)
    except ValueError:
        span = None
    if not sep or span is None or not 0 <= span[0] <= span[1]:
        raise argparse.ArgumentTypeError(f"{text!r} isn't LO-HI (map units, e.g. 110-200)")
    return span


def cmd_map(args) -> int:
    """The map's platforms and her position on them (no keys)."""
    from .control.mover import platform_text
    from .localization import mapdata
    from .localization.pose import Localizer
    model = mapdata.load(args.map)
    marks = []
    if not args.offline:
        client.check_alarm()  # a look brings the game to the front: not while the Lie Detector alarm is up
        eyes = _eyes(model, _game_title())
        s = eyes.look()
        if s.dot:
            w = Localizer(model, geo=s.geo).where_dot(s.dot)  # (the dot's px are the look's window's)
            print(f"she's at map ({w.x:.0f}, {w.y:.0f}) on {platform_text(w.platform)} (dot {s.dot[0]:.1f},{s.dot[1]:.1f})")
            marks.append((w.x, w.y, "her"))
    print(f"{model.name}: {len(model.platforms)} platforms, {len(model.ropes)} ropes; monster spawns by platform:",
          dict(sorted(model.spawn_platforms.items())))
    path = _shot(f"map_{args.map}.png")
    mapdata.render(model, path, marks=marks)
    print(path)
    return 0


def _ways(args) -> None:
    """`--ways N`: the ways each trip of the command chooses among (route.ROUTES; README: Route diversity)."""
    if getattr(args, "ways", None) is not None:
        from .motion import route
        route.ROUTES = args.ways


def cmd_nav(args) -> int:
    _ways(args)
    hunter = _hunter(args)
    nav = hunter.navigator(fighting=args.fight)
    if args.avoid:  # those kinds of move cost 20 times as much: any other way wins
        weight = nav.weight
        nav.weight = lambda m: weight(m) * (20.0 if m.kind in args.avoid else 1.0)
    _record("nav", hunter.model.id)
    if not _to_game():
        return 1
    start = time.monotonic()
    from .behavior.retreat import on_alarm
    with on_alarm(hunter):  # the Lie Detector seen: she retreats to a safe place, then the command stops
        trip = nav.go(args.platform, x=args.x, seconds=args.seconds)
    print(f"{trip.result} in {time.monotonic() - start:.1f} s: {len(trip.moves)} moves in {trip.chains} chains, "
          f"{trip.misses} landed somewhere else, {trip.hits} hit on the way, {trip.waits} waits for monsters, "
          f"{trip.kills} monsters shot dead on the way; time looking {trip.time['look']:.1f} s, "
          f"thinking {trip.time['think']:.1f} s, moving {trip.time['move']:.1f} s")
    for kind, v in sorted(nav.tracker.speeds.items()):
        v = sorted(v)
        print(f"  {kind} speeds seen: median {v[len(v) // 2]:.0f}, 90% {v[int(len(v) * 0.9)]:.0f} units/s ({len(v)})")
    return 0


def cmd_tour(args) -> int:
    from .behavior.tour import Tour
    _ways(args)
    hunter = _hunter(args, low_hp=args.low_hp, potion_hp=args.potion_hp, skill=args.skill)
    nav = hunter.navigator(fighting=True, rng=random.Random(args.seed))  # the route choices too, as the simulator does
    _record("tour", hunter.model.id)
    if not _to_game():
        return 1
    tour = Tour(hunter, nav, log=_log, rng=random.Random(args.seed), rest_below=args.rest_below,
                rest_avoid=args.rest_avoid)
    kinds = set(args.kinds.split(",")) if args.kinds else None
    from .behavior.retreat import on_alarm
    if args.loop:  # the hunt loop (gamelab/docs/hunt-modes.md): one decision per look, for --minutes
        from .behavior import hunt
        if kinds is not None:
            tour.worth = kinds
        if args.ground_tie is not None:  # grounds about as good: the one she left longest ago (README: Route diversity)
            hunt.GROUND_TIE = args.ground_tie
        with on_alarm(hunter):  # the Lie Detector seen: she retreats to a safe place, then the command stops
            if args.grind:  # going round the map as a player grinds (behavior.grind; README: Grinding)
                from .behavior.grind import Grind
                report = Grind(tour, log=_log, sweep=args.sweep).run(minutes=args.minutes)
            else:
                report = hunt.Hunt(tour, log=_log, sweep=args.sweep).run(minutes=args.minutes)
        print(f"hunt done: {report}")
        return 0
    with on_alarm(hunter):
        report = tour.run(stops=args.stops, kinds=kinds, kills_per_stop=args.kills, seconds_per_stop=args.seconds)
    print(f"tour done: {report}")
    return 0


# --- the mini-map and the simulator ----------------------------------------------------------------------

def _map_views(args, width: int):
    """The mini-map's map views, `width` px wide, one per map, made when first needed: `view_for(map id)`
    (None: --map)."""
    from .localization import mapdata
    from .ui.render import MapView
    views: dict = {}

    def view_for(map_id):
        map_id = map_id or args.map
        if map_id not in views:
            views[map_id] = MapView(mapdata.load(map_id), width=width)
        return views[map_id]
    return view_for


def _map_source(args):
    """What the mini-map shows: (source, live): the newest run live (the default), a recorded one
    replayed (--replay), or a route planned offline (--plan)."""
    from .localization import mapdata
    from .ui.sources import LiveSource, ReplaySource, load, planned
    if args.plan:
        src, dst = args.plan
        model = mapdata.load(args.map)
        p = model.platform(src)
        x = args.x if args.x is not None else (p.left + p.right) / 2
        return ReplaySource(planned(model, src, x, dst)), False
    if args.replay:
        return ReplaySource(load(Path(args.replay), args.map), speed=args.speed), False
    return LiveSource(), True


def cmd_minimap(args) -> int:
    """The mini-map: the window beside the game, and the same as a web page for a phone on the same Wi-Fi
    (--window-only or --web-only: just one). Both follow the newest run live (the default), replay a
    recorded one (--replay; each with its own controls), or show a route planned offline (--plan). Both
    only read telemetry: they send no keys and never touch the game."""
    from .ui import web
    server = None
    if not args.window_only:
        source, live = _map_source(args)
        feed = web.Feed(_map_views(args, args.web_width), source, live)
        ends = "Ctrl+C stops it" if args.web_only else "closing the window stops it too"
        try:
            server = web.start(feed, host=args.host, port=args.port, ends=ends)
        except OSError as e:  # the port taken (another mini-map already serving it?)
            print(f"the web page can't start on port {args.port}: {e}")
            if args.web_only:
                return 1
            print("the window only, then (--port picks another one)")
    if args.web_only:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            web.stop(server)
        return 0
    from .ui.app import MapWindow
    view_for = _map_views(args, args.width)
    source, live = _map_source(args)
    position = _beside_game(*view_for(None).size) if live else None
    try:
        MapWindow(view_for, source, live, position=position).run()
    finally:
        if server is not None:
            web.stop(server)
    return 0


def _beside_game(width: int, height: int) -> tuple[int, int] | None:
    """Where the live mini-map goes: beside the game's client area (right, left, else below), on a
    monitor, and never over it: screen captures see whatever window is on top. None if the game isn't
    running or is minimized, or there's no such room (then the window manager places it, and it isn't
    kept on top)."""
    import mss
    try:
        area = window.client_rect(window.find_game(_game_title()))
    except RuntimeError:
        return None
    if area["width"] <= 0 or area["height"] <= 0:
        return None
    game = (area["left"], area["top"], area["left"] + area["width"], area["top"] + area["height"])
    with mss.MSS() as sct:
        monitors = sct.monitors[1:]
    gap = 16  # the window's own frame and title bar fit in this
    for x, y in ((game[2] + gap, game[1]), (game[0] - width - gap, game[1]), (game[0], game[3] + 2 * gap)):
        box = (x - gap, y - 2 * gap, x + width + gap, y + height + gap)
        on_screen = any(m["left"] <= box[0] and box[2] <= m["left"] + m["width"]
                        and m["top"] <= box[1] and box[3] <= m["top"] + m["height"] for m in monitors)
        over_game = box[0] < game[2] and game[0] < box[2] and box[1] < game[3] and game[1] < box[3]
        if on_screen and not over_game:
            return x, y
    print("no room beside the game for the mini-map: keep it off the game window (captures see what's on top)")
    return None


def cmd_sim(args) -> int:
    """Run the stack offline on the simulator (gamelab.sim), recorded for the mini-map. No game, no keys."""
    from .sim import scenarios
    if args.scenario == "list" or args.scenario not in scenarios.SCENARIOS:
        if args.scenario != "list":
            print(f"no scenario {args.scenario!r}; they are:")
        for name, sc in scenarios.SCENARIOS.items():
            print(f"  {name:28s} {sc.doc.splitlines()[0] if sc.doc else ''}")
        return 0 if args.scenario == "list" else 1
    path = telemetry.record_to(f"sim-{args.scenario}")
    print(f"telemetry: {path} (watch it with `python -m gamelab minimap`: a window, and a page for a phone)")
    try:
        trace = scenarios.run_scenario(scenarios.SCENARIOS[args.scenario], pace=args.speed or None)
    finally:
        telemetry.stop()
    for line in trace["log"][-15:]:
        print(" ", line)
    print(f"{args.scenario}: {trace['result']} ({trace['seconds']:.0f} s simulated, {len(trace['plans'])} key plans)")
    return 0


# --- measuring moves in recordings -----------------------------------------------------------------------

def cmd_lock(args) -> int:
    """Press an attack, then jump `d` ms later: from when does the game accept the jump again?"""
    cfg, keys, _ = her_setup()
    attack = getattr(keys, args.action)
    if attack is None:
        print(f"{args.action} isn't bound in config.toml")
        return 1
    window.bring_to_front(cfg.game_window_title)
    cal, crop = _recording_setup()  # she jumps in place: where she is at the start does for every run
    for d in args.delays:
        events = _tap(attack.name, 0) + _tap(keys.jump.name, d)
        rec = recordings.record(events, name=f"lock_{args.action}_{d}", duration_ms=d + 900, crop=crop, scale=cal.scale)
        jumped = [s for s in airborne(track(rec.frames, rec.times, cal.template), geo=rec.geo) if s[0] >= d]
        print(f"jump {d:4d} ms after {args.action}: " +
              (f"airborne at {jumped[0][0]:.0f} ms" if jumped else "IGNORED (still locked)"))
        time.sleep(SETTLE_S)
    return 0


def cmd_combo(args) -> int:
    """Play a combo (optionally several back to back) exactly as the helper would, and check each one."""
    cfg, keys, timing = her_setup(regular_attack=args.regular_attack)
    window.bring_to_front(cfg.game_window_title)
    cal, crop = _recording_setup()  # the runs alternate their facing, so she ends about where she started
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
        rec = recordings.record(events, name=f"{args.name}_{run}", duration_ms=end + 600, crop=crop, scale=cal.scale)
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
        time.sleep(SETTLE_S)
    print(f"\njumps: {totals['back']} back, {totals['up']} straight up, {totals['forward']} forward; "
          f"throws: {totals['front']} toward front, {totals['wrong']} wrong way")
    return 0


def _around_her(frame: np.ndarray, x: int, y: int, geo=None) -> np.ndarray:
    """The AROUND_HER box of a frame, around a name tag whose top-left is (x, y), in a recording of geometry `geo`
    (AROUND_HER is in half-size px of the 1920x1080 window). Cut at the frame's edges, it is padded with black at
    its right and bottom."""
    dx, dy, w, h = AROUND_HER if geo is None else (geo.ifpx(v) for v in AROUND_HER)
    out = np.zeros((h, w, 3), np.uint8)
    src = frame[max(0, y + dy):y + dy + h, max(0, x + dx):x + dx + w]
    out[:src.shape[0], :src.shape[1]] = src
    return out


def cmd_sheet(args) -> int:
    if Path(args.recording).is_dir():
        return _folder_sheet(args, Path(args.recording))
    rec = recordings.load(args.recording)
    keep = [i for i, t in enumerate(rec.times) if args.start <= t <= args.end][::args.every or 3]
    frames, times = rec.frames[keep], rec.times[keep]
    if args.around_her:
        rows = track(frames, times, _calibration(rec.geo).template)
        frames = np.stack([_around_her(f, x, y, rec.geo) for f, (_, x, y, _) in zip(frames, rows)])
    path = _shot(f"{args.recording}_sheet.png")
    recordings.contact_sheet(frames, times, path, cols=args.cols, every=1,
                             zoom=AROUND_HER_ZOOM if args.around_her else 1)
    print(path)
    return 0


def _folder_sheet(args, folder: Path) -> int:
    """A recording kept as JPEG frames with an index (save-recent's, its part before a Lie Detector, or a Lie
    Detector's own): a contact sheet of a frame every --every (by default a second's at 10 a second), --shrink times
    smaller, from --start to --end s; or with --video, all of it as an mp4 playing in real time. Saved in its folder."""
    if args.around_her:
        print("error: --around-her is for the recordings saved by name (her name tag's crop)")
        return 1
    rec = recordings.load_folder(folder)
    if args.video:
        path = folder / "video.mp4"
        n = recordings.folder_video(rec, path, start=args.start, end=args.end)
        print(f"{n} frames, {n / recordings.VIDEO_FPS:.0f} s in real time: {path}")
    else:
        path = folder / "sheet.png"
        n = recordings.folder_sheet(rec, path, start=args.start, end=args.end,
                                    every=args.every or recordings.SHEET_EVERY,
                                    shrink=args.shrink or recordings.SHEET_SHRINK, cols=args.cols)
        print(f"{n} of its {len(rec.frames)} frames: {path}")
    return 0


def cmd_track(args) -> int:
    """Her path through a recording: screen position, height above the start, world x, and jumps."""
    from .analysis.scroll import world_x
    rec = recordings.load(args.recording)
    cal = _calibration(rec.geo)
    rows = track(rec.frames, rec.times, cal.template)
    xs = world_x(rec.frames, rows, rec.geo)
    rest_y = rows[0][2]
    print("   t ms   screen x,y   height   world x   match")
    for i in range(0, len(rows), args.every):
        t, x, y, score = rows[i]
        print(f"  {t:6.0f}   {x:4d},{y:4d}   {rest_y - y:5d}   {xs[i]:7.1f}   {score:.2f}")
    print("airborne:", [(round(a), round(b) if b else None) for a, b in airborne(rows, geo=rec.geo)])
    print(f"net world dx {xs[-1] - xs[0]:+.1f} px (recording scale {cal.scale})")
    return 0


def cmd_check(args) -> int:
    rec = recordings.load(args.recording)
    lag = analyze.send_lag(rec)
    print(f"keys: {lag['sent']}/{lag['planned']} sent, late by at most {lag['max_ms']:.1f} ms (mean {lag['mean_ms']:.1f})")
    pace = analyze.frame_pacing(rec)
    print(f"game frames: {pace['frames']}, gap median {pace['median_ms']} ms, p95 {pace['p95_ms']} ms, "
          f"max {pace['max_ms']} ms; gaps over {analyze.LONG_GAP_MS} ms (start, length): {pace['long_gaps']}")
    print("(long gaps while she stands still are just frames that didn't change)")
    return 0


def cmd_list(args) -> int:
    for path in sorted(RECORDINGS.glob("*.npz"), key=lambda p: p.stat().st_mtime):
        print(path.stem)
    return 0


# --- timing perception ------------------------------------------------------------------------------------

def cmd_bench_look(args) -> int:
    """A look's stages timed on saved frames, or (--live) on the game as it is on screen. It sends no keys
    and never brings the game to the front."""
    import cv2

    from . import capture
    from .analysis import bench
    from .attack import reach
    from .localization import mapdata
    from .localization.senses import Senses
    from .perception.sprites import detector_for
    title = _game_title()
    if args.live and window.foreground_title() != title:
        print("the game isn't in front: bring it forward yourself and run this again (it never does)")
        return 1
    model = mapdata.load(args.map)
    frames = None
    if args.live:
        geo = _game_geometry(title)
        if args.look_scale:
            geo = geometry(*geo.client, args.look_scale)
        eyes = bench.FrontEyes(title, _calibration(geo).template, detector_for(model, geo),
                               screen=capture.Grabber(args.capture), geo=geo,
                               minimap_picture=mapdata.minimap_picture(model.id))
    else:
        files = [Path(f) for f in args.files] or bench.saved_frames()
        if not files:
            print(f"no frames: name some saved screenshots (1920x1080 or 1366x768; the default ones are in {LAB})")
            return 1
        frames = bench.load(files, tuple(int(v) for v in args.shrink.lower().split("x")) if args.shrink else None)
        geo = bench.geometry_of(frames, args.look_scale)
        eyes = bench.SavedEyes(_calibration(geo).template, detector_for(model, geo), geo,
                               minimap_picture=mapdata.minimap_picture(model.id), check_frames=not args.shrink)
        if args.shrink:
            print("--shrink: the frames' own checks are left out (perception.framecheck: the shrink blurs their 1-px "
                  "anchors)")
        if bad := bench.refused(eyes, Senses(eyes, model, band=(reach.LOOK_UP, reach.LOOK_DOWN)), frames):
            print(f"frames whose looks fail their checks (--map {model.id}?):" + "".join(f"\n  {b}" for b in bad))
            return 1
    from .localization.worldstate import Asks, Schedule, WorldStep
    senses = Senses(eyes, model, band=(reach.LOOK_UP, reach.LOOK_DOWN))
    # The continuous feed's step on the same frames (localization.worldstate): her place and every monster it
    # follows re-found, and with what a frame may also be asked for. Each frame is looked at several times in
    # a row, as a still scene; the whole-screen one finds the monsters the others then re-find.
    step = WorldStep(Senses(eyes, model, band=(reach.LOOK_UP, reach.LOOK_DOWN)),
                     schedule=Schedule(band_s=None, wide_s=None))

    def frame_step(asks=None):  # the searches asked for at once, on this thread, and merged (InlineSearches)
        full = eyes._capture()
        st = step.step(lambda ropes: eyes.look_at(full, None, ropes), time.monotonic(), True, asks)
        step.came_back()
        return st
    # A look with the sweep on (Senses.sweep, as the hunt loop's are): her band and a slice of the rest of the screen
    # in turn, the slices going round from look to look, and the frames with them.
    swept = Senses(eyes, model, band=(reach.LOOK_UP, reach.LOOK_DOWN), sweep=True)
    looks = {"Senses.look: no monsters": lambda: senses.look(mobs=False),
             "Senses.look: her band": lambda: senses.look(),
             "Senses.look: band + loot on her platform": lambda: senses.look(loot=True),
             "Senses.look: ... and a slice of the sweep": lambda: swept.look(loot=True),
             "Senses.look: whole screen + all loot": lambda: senses.look(wide=True, loot=True),
             "feed step: her place, monsters re-found": frame_step,
             "feed step: ... and her band searched": lambda: frame_step(Asks(band=True)),
             "feed step: ... and the whole screen": lambda: frame_step(Asks(wide=True))}
    timer = bench.run(eyes, senses.band, looks, frames, reps=args.reps, n=args.n)
    timer.count("feed step: monsters re-found", step.counts["re-found"])
    timer.count("... of the re-finds", step.counts["re-finds"])
    timer.count("... that tried the kind's whole set", step.counts["fallbacks"])
    from . import native
    where = f"{args.n} live captures" if frames is None else f"{len(frames)} saved frames x {args.reps}"
    w, h = geo.frame_size
    print(f"{where} of a {geo.client[0]}x{geo.client[1]} client, looks at {w}x{h}; "
          f"OpenCV {cv2.__version__}, {cv2.getNumThreads()} threads"
          + ("" if frames is None else "; the grab itself (16.7 ms with mss, about 2 with DXGI) isn't included"))
    print("the C pieces (gamelab/native.c): " + ("in use" if native.lib() is not None
                                                  else f"not in use ({native.why}): numpy instead"))
    print("\n".join(timer.table()))
    if frames is None:
        print(eyes.screen.summary())
    return 0


NATIVE_FRAMES = REPO / "tests" / "fixtures" / "native"  # bench-paste's frames of the game at 1366x768 by default


def cmd_bench_paste(args) -> int:
    """A map's monsters (each kind with a colour gate) and its drops (perception.drops), the game's own pictures drawn
    onto frames of the game at its own 1366x768 as the game draws them, looked for at that size and on the same scenes
    stretched to 1920x1080 (analysis/pasted.py). Offline: no game, no keys."""
    from .analysis import pasted
    from .attack import reach
    from .localization import mapdata
    from .perception import drops, loot, sprites
    from .perception.kinds import kind_of
    from .perception.mobs import MobDetector
    from .screen import NATIVE
    files = [Path(f) for f in args.files] or sorted(NATIVE_FRAMES.glob("*.png"))
    backgrounds = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    if not backgrounds or any((b.shape[1], b.shape[0]) != NATIVE.client for b in backgrounds):
        print("bench-paste draws onto frames of the game at its own 1366x768: name some (by default "
              "tests/fixtures/native)")
        return 1
    model = mapdata.load(args.map)
    print(f"{len(backgrounds)} frames of 1366x768; map {args.map}; each picture drawn {args.each} times both ways")
    try:
        native = sprites.detector_for(model, NATIVE)
        detectors = {"1366x768, full size": (native, NATIVE),
                     "1920x1080 (the scene stretched), half size": (sprites.detector_for(model, STRETCHED), STRETCHED)}
        if args.full_step_1:  # step 1 at the look's own size, as before the search went coarse to fine
            detectors["1366x768, full size, step 1 at full size"] = (
                MobDetector(sprites.folder_for(NATIVE.stretch), frame_scale=NATIVE.frame_scale, search_scale=1.0),
                NATIVE)
        kinds = {kind_of(s.name) for s in model.spawns if s.name} & set(native.templates)
    except RuntimeError:  # (no picture of any kind with a colour gate there at all)
        kinds = set()
    if not kinds:
        print(f"monsters: none of map {args.map}'s has a colour gate (mobs.PALETTES), nor its background colours for a "
              f"learned one (learn-colours)")
    else:
        print(f"monsters ({', '.join(sorted(kinds))}):")
        result = pasted.monsters(backgrounds, detectors, sprites.folder_for(NATIVE.stretch),
                                 (reach.LOOK_UP, reach.LOOK_DOWN), args.each, args.seed, kinds)
        print("\n".join("   " + line for line in pasted.lines(result)))
        print("re-finds (each found, then drawn where it stood as a frame it may show next, and looked for again "
              "there as the feed does):")
        result = pasted.refinds(backgrounds, detectors, sprites.folder_for(NATIVE.stretch),
                                (reach.LOOK_UP, reach.LOOK_DOWN), args.seed + 2, kinds)
        print("\n".join("   " + line for line in pasted.refind_lines(result)))
    items = drops.items_for(model)
    found = {"1366x768, full size": (loot.detector_for(model, NATIVE), NATIVE),
             "1920x1080 (the scene stretched), half size": (loot.detector_for(model, STRETCHED), STRETCHED)}
    print(f"drops: {', '.join(items)}, and mesos")
    result = pasted.drops(backgrounds, found, loot.folder_for(NATIVE.stretch), args.each, args.seed + 1,
                          loot.kinds_of(items))
    print("\n".join("   " + line for line in pasted.lines(result, "drops")))
    return 0


def cmd_learn_colours(args) -> int:
    """A map's background colours, from saved frames of it, for the gates of its monsters without a hand-made colour
    test (perception.gates, sprites.learn_background): kept in assets/backgrounds/<map id>.npz (or --out), where the
    map's detector finds them. Offline: no game, no keys."""
    from .localization import mapdata
    from .perception import gates, sprites
    from .perception.kinds import kind_of
    from .perception.mobs import PALETTES
    from .screen import of_frame
    model = mapdata.load(args.map)
    kinds = sorted({kind_of(s.name) for s in model.spawns if s.name} - set(PALETTES))
    if not kinds:
        print(f"map {args.map}'s monsters all have a hand-made colour test: nothing to learn")
        return 0
    files = sorted(f for name in map(Path, args.files) for f in ([name] if name.is_file() else name.glob("*.png")))
    frames = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    if not frames:
        print("name saved frames of the map (full-size screenshots of the game window, or folders of them)")
        return 1
    sizes = sorted({f"{f.shape[1]}x{f.shape[0]}" for f in frames})
    print(f"map {args.map} ({', '.join(kinds)}): {len(frames)} frames ({', '.join(sizes)})")
    counts, cut = sprites.learn_background(frames, lambda geo, back: sprites.detector_for(model, geo, back),
                                           args.rounds, log=lambda line: print("   " + line))
    path = gates.save_background(args.map, counts, len(frames), cut, Path(args.out) if args.out else gates.BACKGROUNDS)
    print(f"kept {path} ({int(counts.sum()):,} pixels at step 1's scale)")
    for geo in sorted({of_frame(f) for f in frames}, key=lambda g: g.client):
        det = sprites.detector_for(model, geo, counts)
        print(f"   {geo.client[0]}x{geo.client[1]}: " + "; ".join(
            f"{k}'s gate takes {100 * det.gates.share(k, counts):.2f}% of the background" for k in sorted(det.learned))
              + "".join(f"; {', '.join(kinds)} together {100 * det.gates.share(kinds, counts):.2f}%"
                        for kinds, _, _ in det.units if len(kinds) > 1))
    return 0


def cmd_bench_capture(args) -> int:
    """A look's capture and a glance's timed through DXGI and mss, on the game's window as it is on screen
    (or a 1920x1080 box at the screen's top left if the game isn't running). Reads the screen only: no
    keys, and nothing is brought to the front."""
    from . import capture
    from .analysis import bench
    try:
        area = window.client_rect(window.find_game(_game_title()))
        what = "the game's window as it is on screen"
    except RuntimeError:
        area = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        what = "the screen's top-left 1920x1080 (the game isn't running)"
    if area["width"] <= 0 or area["height"] <= 0:
        print("the game is minimized: restore it yourself and run this again")
        return 1
    dxgi, gdi = capture.Grabber("dxgi"), capture.Grabber("mss")
    timer = bench.Timer()
    bench.capture_stages(area, dxgi, gdi, bench.Timer())  # the first duplicates the monitor: not timed
    for _ in range(args.n):
        bench.capture_stages(area, dxgi, gdi, timer)
    print(f"{args.n} rounds on {what}")
    print("\n".join(timer.table()))
    print(dxgi.summary())
    return 0


BODY_RECORDINGS = ("stand_snail", "walk_snail")  # she stands still while snails walk: the camera is still


def cmd_heading(args) -> int:
    """Which way she faces on saved frames (perception.heading: her picture above her name tag against her pictures),
    and with --learn, frames whose facing you know teach her pictures (lab, or with --seed the seed in gamelab/assets
    that stands in until she has her own). Her tag is found as a look finds it, or with --tags where a study put it (a
    JSON list of {"file", "tag": [x, y]}, look-frame px: a window's own "BugBaby" label, Character Info's, can take
    the search). Offline: no game, no keys."""
    import json

    import cv2

    from .perception.heading import Heading
    from .perception.nametag import NameTag
    from .screen import of_frame
    files = [Path(f) for f in args.frames]
    if not files:
        print("name some saved frames (PNG, the game's whole client area)")
        return 1
    fulls = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    geo = of_frame(fulls[0])
    if any(of_frame(f) != geo for f in fulls):
        print("the frames aren't all of one size")
        return 1
    template, full_path = _calibration(geo).template, paths.calibrated(paths.TAG_FULL, geo.suffix)
    lab, seed = paths.calibrated(paths.HEADING, geo.suffix), paths.calibrated(paths.HEADING_SEED, geo.suffix)
    bank = Heading(geo, seed if args.seed else lab, None if args.seed else seed, background=False, log=print)
    placed = ({Path(o["file"]).name: tuple(o["tag"]) for o in json.loads(Path(args.tags).read_text())}
              if args.tags else {})
    if args.fresh:
        bank.forget()
    print(f"her pictures: {len(bank.refs)} from {len(bank.pool)} boxes ({bank.path if bank.path.exists() else seed})")
    added = 0
    for f, full in zip(files, fulls):
        k = geo.look_scale
        frame = full if k == 1 else cv2.resize(full, (full.shape[1] // k, full.shape[0] // k),
                                               interpolation=cv2.INTER_AREA)
        # (a tag of its own each: frames of different places)
        at = placed.get(f.name) or NameTag(template, full_path, geo).find(full)
        r = bank.read(frame, at)
        print(f"{f.name}: tag {at}; " + (f"faces {r.side}" if r.side else f"can't tell ({r.why})")
              + (f" (matched {r.score:.2f}, {r.margin:.2f} ahead of the other way)" if r.why not in
                 ("no pictures", "no tag", "off the frame", "minimap") else ""))
        if args.learn and at is not None:
            added += bank.add(frame, at, args.learn)
    if args.learn:
        bank.rebuild()
        print(f"learnt {added} of {len(files)} frames as facing {args.learn}: {len(bank.refs)} pictures from "
              f"{len(bank.pool)} boxes, kept at {bank.path}")
    return 0


def cmd_bodies(args) -> int:
    """Where the monster detector puts monsters: their feet against the platform's surface on saved
    screenshots, and their x from frame to frame on recordings (analysis/bodies.py). Offline: no game, no
    keys."""
    from .analysis import bench, bodies
    from .attack import reach
    from .localization import mapdata
    from .localization.senses import Senses
    from .perception.sprites import detector_for
    if args.tracks:  # the feed's tracks in `watch`'s telemetry
        for f in args.tracks:
            print(f"{Path(f).name}:")
            for line in bodies.tracks_lines(telemetry.read(Path(f))):
                print("   " + line)
        return 0
    model = mapdata.load(args.map)
    files = [Path(f) for f in args.files] or bench.saved_frames()
    frames = bench.load(files)
    geo = bench.geometry_of(frames)
    detector = detector_for(model, geo)
    eyes = bench.SavedEyes(_calibration(geo).template, detector, geo, minimap_picture=mapdata.minimap_picture(model.id))
    if bad := bench.refused(eyes, Senses(eyes, model, band=(reach.LOOK_UP, reach.LOOK_DOWN)), frames):
        print(f"frames whose looks fail their checks (--map {model.id}?):" + "".join(f"\n  {b}" for b in bad))
        return 1
    senses = Senses(eyes, model, band=(reach.LOOK_UP, reach.LOOK_DOWN))

    def look_at(bgra):
        def look():
            eyes.show(bgra)
            return senses.look(wide=True)
        return look

    scenes = [(name, look_at(bgra)) for name, bgra in frames]
    names = BODY_RECORDINGS if args.recordings is None else args.recordings
    found = [Path(r) if Path(r).suffix else RECORDINGS / f"{r}.npz" for r in names]
    detectors = {geo: detector}  # the recordings', by their own geometry (the default ones: 1920x1080 half size)

    def for_recording(g):
        if g not in detectors:
            detectors[g] = detector_for(model, g)
        return detectors[g]
    print("\n".join(bodies.run(scenes, model, [p for p in found if p.exists()], for_recording)))
    if missing := [str(p) for p in found if not p.exists()]:
        print(f"recordings not found: {', '.join(missing)}")
    return 0


# --- watching the game, no command running -------------------------------------------------------------

def _watch_runner(args):
    """What `watch` looks with: the game on screen (never brought to the front), her throws unknown (played
    by hand), drops searched on a schedule of their own."""
    from .attack import reach
    from .localization import feed, worldstate
    title = _game_title()
    geo = _game_geometry(title)  # (minimized, it waits for the game to be restored: it never restores it)
    return feed.screen_world(args.map, title, _calibration(geo).template, (reach.LOOK_UP, reach.LOOK_DOWN),
                             throws=False, schedule=worldstate.WATCH, geo=geo)


def cmd_watch(args) -> int:
    """The game looked at every frame with no command running, as `--feed` does for one: her place, the
    monsters tracked (ids, facing, HP bars, deaths and who killed them), her EXP gains and hits, published as
    telemetry for the mini-map. It never sends a key and never brings the game to the front: a frame the
    game isn't wholly showing isn't read. With --lie-detector it also checks the screen for the Lie Detector
    and pages whoever is at the PC (the player keeps its own watchdog regardless): on a thread of its own, the
    frame handed over as soon as it's taken, before its step (it had waited for the step, in the frame loop). With
    --heading it reads which way she faces off every frame (perception.heading, her pictures in lab or the seed),
    says each change, and publishes it with the scenes (her `facing_seen`): turn her with the arrows and watch."""
    from .localization import feed, worldstate
    from .perception.drops import unknown_note
    from .perception.lie_detector import LieDetectorSpotter
    feed.quiet()  # below normal priority, as the feed's child runs
    runner = _watch_runner(args)
    model, band = runner.step.model, runner.step.senses.band
    path = telemetry.record_to("watch", telemetry.FOLDER)
    telemetry.emit("session", map=model.id, command="watch", mode="watch")
    print(f"watching the game {args.hz:.0f} times a second; telemetry: {path} (the mini-map follows it); Ctrl+C stops")
    if note := unknown_note(model):  # its monsters whose drops aren't known
        print(note)
    spotter = LieDetectorSpotter() if args.lie_detector else None
    started = time.monotonic()
    until = started + args.minutes * 60 if args.minutes else None
    w = {"scene": -1e9, "tracks": -1e9, "status": started, "lie": -1e9, "visible": None, "loot": [], "looked": [],
         "frames": 0, "seen": 0, "alarm": False, "facing": "", "told": 0, "read": 0}
    if args.heading:
        bank = runner.eyes.heading_bank()
        print(f"her facing off every frame: {len(bank.refs)} pictures of her from {len(bank.pool)} boxes"
              + ("" if bank.ready else " (none: nothing can be told; `heading --learn` makes some from saved frames)"))

    def facing(st: worldstate.WorldState):
        """Which way the frame shows her facing (--heading), each change said."""
        r = runner.eyes.heading(SimpleNamespace(frame=st.frame, tag=st.tag, tag_seen=st.tag_seen, geo=st.geo, t=st.t))
        w["read"] += 1
        w["told"] += r.side is not None
        said = r.side or f"can't tell ({r.why})"
        if said != w["facing"]:
            w["facing"] = said
            _log(f"  her frame: faces {r.side} (matched {r.score:.2f}, {r.margin:.2f} ahead)" if r.side else
                 f"  her frame: {said}")
        return r

    def check(full) -> None:  # the Lie Detector, on its thread
        if spotter.seen(full) and not w["alarm"]:
            w["alarm"] = True
            _lie_detector_seen(full)
    lie = worldstate.Worker(check, "lie-detector") if spotter is not None else None

    def shown(frame) -> None:
        now = time.monotonic()
        if lie is None or frame.full is None or now - w["lie"] < WATCH_LIE_DETECTOR_S:
            return
        lie.take()  # (a check that failed raises here)
        if lie.free():
            w["lie"] = now
            lie.submit(frame.full)

    def emit(st: worldstate.WorldState, frame) -> None:
        now = time.monotonic()
        if st.late:  # a frame searched, given again with what its searches found: its drops, its events
            if st.loot is not None:  # (and where it looked for them, for the scene records: Scene.record's `looked`)
                w["loot"], w["looked"] = st.loot, st.looked
            for e in st.events:
                if e.kind in WATCH_EVENTS:
                    telemetry.emit("event", kind=e.kind, text=e.text(), track=e.track)
            return
        w["frames"] += 1
        if st.visible != w["visible"]:
            w["visible"] = st.visible
            _log("the game is on screen: watching" if st.visible else f"not watching: {st.why} (nothing is read)")
        if st.visible:
            w["seen"] += 1
            for e in st.events:
                if e.kind in WATCH_EVENTS or (e.kind == "hp_drop" and e.data.get("new")):
                    telemetry.emit("event", kind=e.kind, text=e.text(), track=e.track)
                if e.kind == "died":
                    by = e.data.get("by")  # the other player by it (localization.players)
                    _log(f"  {e.data['kind']} #{e.track} died on platform {e.data['platform']}: "
                         + ("hers" if e.data["cause"] == "hers" else "someone else's"
                            + (f" (another player's, #{by})" if by is not None else "")))
            if st.loot is not None:
                w["loot"], w["looked"] = st.loot, st.looked
            seen = facing(st) if args.heading else None
            if now - w["scene"] >= WATCH_SCENE_S:
                w["scene"] = now
                s = worldstate.to_scene(st, model, band, mobs=True, wide=True)
                s.loot, s.looked = w["loot"], w["looked"]  # as of the last frame searched for drops (LOOT_EVERY_S)
                record = s.record()
                if seen is not None and record["her"] is not None:
                    record["her"].update(facing_seen=seen.side, facing_strong=seen.strong)
                telemetry.emit("scene", **record)
            if now - w["tracks"] >= WATCH_TRACKS_S:
                w["tracks"] = now
                telemetry.emit("tracks", list=[tr.record() for tr in st.tracks], frames=st.seq)
        meter = runner.step.meter  # her EXP as read off its number (localization.gains)
        while runner.step.gains:  # each gain as the hunter records it (Hunter._gained), for the mini-map's EXP line
            g = runner.step.gains.popleft()
            if meter.at is not None:
                rate = meter.per_minute(now)
                telemetry.emit("exp", exp=meter.at[1], level=meter.at[0], gain=g.exp, kinds=list(g.kinds),
                               total=meter.total, per_minute=None if rate is None else round(rate, 1))
        if now - w["status"] >= WATCH_STATUS_S:
            c, rate = runner.step.tracker.counts, w["frames"] / max(1e-9, now - w["status"])
            live = [tr for tr in st.tracks if tr.state != "dead"]
            for note in meter.notes:  # (its sizes going off, or on again)
                _log(f"  {note}")
            meter.notes.clear()
            per_minute = meter.per_minute(now)
            exp = "" if meter.at is None else f"; EXP {meter.at[1]:,}" + (
                "" if per_minute is None else f", {per_minute:.0f} a minute")
            told = f"; her facing told on {w['told']} of {w['read']} frames" if args.heading else ""
            print(f"{time.strftime('%H:%M:%S')} {rate:.1f} frames a second ({w['seen']} with the game on screen); "
                  f"{len(live)} monsters followed; deaths seen: {c['died hers']} hers, {c['died other']} others'{exp}"
                  f"{told}", flush=True)
            w["status"], w["frames"], w["seen"], w["told"], w["read"] = now, 0, 0, 0, 0

    # Nobody waits for watch's searches (each is merged at the next frame), so its wait for the next frame needn't
    # end when one is back: time.sleep's, on Windows' fine timer, not an Event's, which ends on its 15.6 ms ticks.
    def poll(wait: float):
        time.sleep(wait)
        return None, False

    try:
        feed.run(runner, args.hz, poll, emit, until, shown)
    except KeyboardInterrupt:
        pass
    finally:
        runner.step.close()
        if lie is not None:
            lie.close()
        telemetry.stop()
    print(f"watched {time.monotonic() - started:.0f} s: {runner.step.counts['frames']} frames "
          f"({runner.step.counts['not visible']} not showing the game); deaths {runner.step.tracker.counts}")
    return 0


def _lie_detector_seen(frame) -> None:
    """The Lie Detector seen while watching: the alarm (gamelab refuses to send keys until it's cleared) and a
    page, once; nothing is pressed."""
    from .control import alarm
    _log("LIE DETECTOR on screen: answer it yourself; gamelab won't send keys until `clear-alarm`")
    if alarm.alarm_info() is None:
        alarm.raise_alarm(frame, "lie_detector (seen by watch)")


# --- the rolling recording -------------------------------------------------------------------------------

def cmd_recorder(args) -> int:
    """The rolling recording (control.rolling): the game's window 10 times a second, the last 5 minutes kept in memory,
    saved on a Lie Detector alarm and by `save-recent`; until Ctrl+C. Reads the screen only: no keys, and the game is
    never brought forward. With --status, how the running one is doing."""
    from .control import rolling
    if args.status:
        print("\n".join(rolling.status_lines()))
        return 0
    lock = rolling.only_one()  # held until the process ends
    if lock is None:
        print("a recorder is running already (one at a time): `recorder --status` says how it's doing")
        return 0
    rolling.quiet()
    rec = rolling.Recorder(rolling.GameWindow(_game_title()), rolling.ScreenGrab(), log=lambda m: print(m, flush=True))
    rec.run(args.minutes * 60 if args.minutes else None)
    return 0


def cmd_save_recent(args) -> int:
    """The running recorder's last --seconds saved to gamelab/lab/recent/<time>[-name]/ (frames, index, summary): it's
    asked through its inbox, and the folder is printed last."""
    from .analysis import disk
    from .control import rolling
    try:
        out = rolling.ask(args.seconds, args.name)
    except rolling.NotRunning as e:
        print(f"error: {e}")
        return 1
    if "error" in out:
        print(f"error: the recorder didn't save: {out['error']}")
        return 1
    gaps = ", ".join(f"{why} {s:g} s" for why, s in out["gaps"].items())
    window_size = "" if out["window"] is None else f", the window {out['window'][0]}x{out['window'][1]}"
    print(f"saved {out['seconds']:g} s: {out['frames']} frames ({disk.size(out['bytes'])}){window_size}"
          + (f"; no frame: {gaps}" if gaps else "") + f" (`sheet FOLDER`, `sheet FOLDER --video` to look at it)")
    print(out["folder"])
    return 0


# --- what gamelab keeps on disk --------------------------------------------------------------------------

def cmd_lab_usage(args) -> int:
    """What gamelab keeps under gamelab/lab, by kind. It deletes nothing, unless --prune-telemetry: then the
    telemetry a run starting now would delete (the oldest past 2 GB in all; none for its age)."""
    from .analysis import disk
    lab = paths.LAB
    if args.prune_telemetry:
        gone = telemetry.prune(lab / "telemetry")
        print(f"deleted {len(gone)} telemetry files ({disk.size(sum(b for _, b in gone))})")
    print("\n".join(disk.report(lab)))
    return 0


def cmd_coverage(args) -> int:
    """How often each move of the map's routes was planned, taken and missed, where she spent her time, and the
    moves never taken with their cost gap and why they lost (analysis/coverage.py): from saved harness runs and the
    live telemetry, with a picture of the map. Offline: no game, no keys."""
    from .analysis import coverage
    from .localization import mapdata
    from .sim import harness
    from .ui import render
    _ways(args)  # (what the navigator's choice would do on the runs' plans: coverage.drawn)
    model = mapdata.load(args.map)
    graph = coverage.Graph(model)
    sets = []
    for name in args.runs:
        try:
            runs = harness.load(name)
        except FileNotFoundError:
            print(f"{name}: no such saved run in {harness.FOLDER}: left out")
            continue
        if (other := {r.get("map", harness.FIRST_MAP) for r in runs} - {args.map}):
            print(f"{name}: its runs are on map {', '.join(map(str, sorted(other)))}, not {args.map}: left out")
            continue
        tallies = [r["coverage"] for r in runs if "coverage" in r]
        if not tallies:
            print(f"{name}: its runs keep no coverage (saved before the harness kept it): left out")
            continue
        first = runs[0]
        sets.append((f"harness {name} ({first['policy']}, preset {first['preset']}, {len(tallies)} seeds of "
                     f"{first['minutes']:g} min)", coverage.merge(tallies, graph)))
    if args.live or not args.runs:
        files = [p for p in sorted(paths.TELEMETRY.glob("*.jsonl")) if "-sim-" not in p.name]
        tallies = []
        for path in files:
            records = telemetry.read(path)
            session = next((r for r in records if r.get("topic") == "session"), {})
            if session.get("map", args.map) == args.map:  # (another map's moves aren't this one's)
                tallies.append(coverage.tally(records, graph))
        sets.append((f"live telemetry ({len(tallies)} files in {paths.TELEMETRY})", coverage.merge(tallies, graph)))
    if not sets:
        return 1
    lines = coverage.report(graph, sets)
    print("\n".join(lines))
    if args.text:
        Path(args.text).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report: {args.text}")
    picture = Path(args.picture) if args.picture else _shot(f"coverage_{args.map}.png")
    for i, (name, t) in enumerate(sets):
        path = picture if i == 0 else picture.with_name(f"{picture.stem}-{i + 1}{picture.suffix}")
        Image.fromarray(render.coverage_picture(model, coverage.picture_rows(graph, t), title=name)).save(path)
        print(f"picture of {name}: {path}")
    return 0


def cmd_respawn(args) -> int:
    """Respawns, drop life and how full the platforms are, from telemetry (analysis/respawn.py): by default the
    live telemetry (but the simulator's files), against the map's own spawn fields. Offline: no game, no keys."""
    import json
    from .analysis import respawn
    from .attack import reach
    from .behavior.probe import PROBES
    from .localization import mapdata
    from .localization.pose import Camera
    model = mapdata.load(args.map)
    corner, band = Camera(model.bounds).corner, (reach.LOOK_UP, reach.LOOK_DOWN)
    files = ([Path(f) for f in args.files] if args.files else
             [p for p in sorted(paths.TELEMETRY.glob("*.jsonl")) if "-sim-" not in p.name])
    runs = respawn.Runs()
    for path in files:
        records = telemetry.read(path)
        session = next((r for r in records if r.get("topic") == "session"), {})
        if session.get("map", args.map) != args.map:
            continue
        probe = PROBES.get(str(session.get("command", "")).removeprefix("probe-"))
        runs.add(records, model, corner, band, probe_wide=probe is not None and probe.wide)
    raw = json.loads((mapdata.CACHE / f"{args.map}.json").read_text(encoding="utf-8"))
    lines = respawn.report(runs, model, respawn.spawn_fields(raw["tree"]))
    print("\n".join(lines))
    if args.text:
        Path(args.text).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report: {args.text}")
    return 0


def cmd_memory(args) -> int:
    """Her memory of the monsters (localization.memory) run again on a recorded run: what it would have held at each
    look, as 'memory' records in a copy of the run (`--out`, for `minimap --replay`), and how it went. Offline: no game,
    no keys."""
    import json
    from collections import Counter
    from .attack import reach
    from .localization import mapdata, memory
    records = telemetry.read(Path(args.file))
    session = next((r for r in records if r.get("topic") == "session"), {})
    model = mapdata.load(args.map or session.get("map") or DEFAULT_MAP)
    kept = memory.Memory(model)
    out = memory.replay(records, model, (reach.LOOK_UP, reach.LOOK_DOWN), memory=kept)
    mem = [r for r in out if r.get("topic") == "memory"]
    looks = sum(1 for r in records if r.get("topic") == "scene" and r.get("searched"))
    held = Counter()
    for r in mem:
        for e in r["list"]:
            if e["state"] == "seen":
                held["sure" if e["p"] >= memory.SURE else "unsure"] += 1
    n = max(1, len(mem))
    print(f"{args.file}: map {model.id}, {looks} looks searched for monsters, {len(mem)} 'memory' records")
    print(f"  monsters held a record: {held['sure'] / n:.2f} sure (p >= {memory.SURE}), {held['unsure'] / n:.2f} less")
    c = kept.counts
    print(f"  {c['new']} monsters seen, {c['killed']} killed by her, {c['gone']} forgotten by looks covering their "
          f"place, {c['aged']} by age")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        print(f"  written: {args.out} (python -m gamelab minimap --replay {args.out})")
    return 0


def cmd_patrol(args) -> int:
    """Grinding's view of a map (behavior.patrol): its stops, the loop it would take and camping's best, before she has
    seen anything; with a recorded run (`file`), its looks run through her memory and the belief, and every `--every` s
    from where she was, the next stop grinding would have taken there and why. Offline: no game, no keys."""
    import re
    from collections import Counter
    from .behavior import patrol
    from .localization import mapdata, memory
    from .localization.players import Player
    from .motion.route import Planner
    from .attack import reach
    from .perception.kinds import kind_of
    records = telemetry.read(Path(args.file)) if args.file else []
    session = next((r for r in records if r.get("topic") == "session"), {})
    model = mapdata.load(args.map or session.get("map") or DEFAULT_MAP)
    exp = {kind_of(n): e for n, e in model.mob_exp.items()}
    best = max(exp.values(), default=0)
    kinds = {k for k, e in exp.items() if e >= 0.25 * best} or None  # tour.WORTH's cut
    planner = Planner(model)
    p = patrol.Patrol(model, kinds, exp, travel=planner.arrive)
    print(f"{model.name}: {len(p.points)} spawn points of {sorted(kinds or ())}, {len(p.stops)} stops")
    for st in p.stops:
        reached = ", ".join(f"{p.points[q].kind} on {p.points[q].platform}" + ("" if c >= 1 else f" ({c:.0%})")
                            for q, c in st.cover)
        print(f"  stop {st.platform} at x {st.x:.0f}: {reached}")
    p.belief.t = 0.0
    p.replan(0.0, force=True)
    camp = p.camping(0.0)
    print(f"loop: {p.plan.text(p.platform_of)}; camping at {p.platform_of(camp.stops[0])}: {camp.rate * 60:.0f}")
    if not records:
        return 0
    kept, kills = memory.Memory(model), []
    kept.kills = lambda: kills
    kill_re = re.compile(r"^(?P<kind>\S+) down on platform (?P<platform>-?\d+)")
    band, last, start, why = (reach.LOOK_UP, reach.LOOK_DOWN), None, None, Counter()
    for r in records:
        topic = r.get("topic")
        if topic == "event" and r.get("kind") == "kill" and (m := kill_re.match(r.get("text", ""))):
            kills.append(memory._Kill(r["t"], m["kind"], int(m["platform"]), float(r.get("x") or 0.0)))
        if topic != "scene":
            continue
        t, her = r["t"], r.get("her") or {}
        start = t if start is None else start
        if not r.get("searched") or her.get("x") is None:
            continue
        players = [Player(q["id"], q["kind"], q["x"], q["y"], q.get("platform"), (0.0, 0.0), t, t)
                   for q in r.get("players", [])]
        sights = [memory._monster(m) for m in [*r.get("monsters", []), *r.get("swept", [])]]
        kept.update(t, sights, memory.cover_of_record(r, band), players)
        p.belief.update(t, kept, players)
        if her.get("platform") is None or (last is not None and t - last < args.every):
            continue
        last = t
        if (changed := p.replan(t)) is not None:
            print(f"{t - start:7.1f}  loop: {p.plan.text(p.platform_of)} ({changed})")
        j, info = p.choose(t, her["platform"], her["x"], current=p.stop_on(her["platform"]))
        if j is not None:
            why[info["why"]] += 1
            print(f"{t - start:7.1f}  on {her['platform']} at x {her['x']:.0f}: next {patrol.choice_text(info)}")
    print(f"choices: {dict(why)}; refills x{p.belief.scale:.2f} of the model ({len(p.belief.refills)} watched); others "
          f"take {({k: round(p.belief.take(k), 2) for k in sorted({q.kind for q in p.points})})}")
    return 0


# --- the command line ------------------------------------------------------------------------------------

def feed_arg(s) -> None:
    s.add_argument("--feed", action="store_true",
                   help="looks and glances from a continuous feed: the game looked at every frame in a process of its "
                        "own (not tried in game yet)")


def sweep_arg(s, default: str) -> None:
    s.add_argument("--sweep", action=argparse.BooleanOptionalAction, default=None,
                   help="each look at the band around her also searches a slice of the rest of the screen in turn, so "
                        f"her memory of the monsters is kept up (localization.senses; default: {default})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gamelab", description="Drive the real game with exact key timelines and measure what happened.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("player", help="run the key player (use --elevate: the game runs as admin)")
    s.add_argument("--elevate", action="store_true", help="start it as administrator via a UAC prompt")
    s.set_defaults(func=cmd_player)
    sub.add_parser("stop", help="stop the player").set_defaults(func=cmd_stop)
    sub.add_parser("clear-alarm", help="after you've answered a Lie Detector and its recording has ended: let "
                                       "gamelab run again (start the player again after it)"
                   ).set_defaults(func=cmd_clear_alarm)
    s = sub.add_parser("spot", help="check the game (or an image) for the Lie Detector; sends no keys")
    s.add_argument("--image", help="check this screenshot instead of the game")
    s.set_defaults(func=cmd_spot)

    s = sub.add_parser("calibrate", help="find her on screen and save her name tag template")
    s.add_argument("--tag", help="name tag centre X,Y in client pixels, if it isn't found automatically")
    s.add_argument("--fresh", action="store_true", help="rebuild the name tag template (stand clear of ropes)")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("window", help="resize the game's client area to 1366x768 (the game's own size: pixel for "
                                      "pixel) or 1920x1080, through the player; sends no keys")
    s.add_argument("size", choices=["1366x768", "1920x1080"])
    s.set_defaults(func=cmd_window)

    s = sub.add_parser("snap", help="save a screenshot of the game to gamelab/lab/shots/snap.png")
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

    s = sub.add_parser("hunt", help="hunting objectives that run on their own (see gamelab/behavior/hunter.py)")
    s.add_argument("mode", choices=["look", "kill", "clear", "shots"],
                   help="look: what she sees now; kill: one monster; clear: everything in reach, nearest "
                        "first; shots: hits and misses so far, by target height (sends no keys)")
    s.add_argument("--kinds", help="only these monsters, e.g. slime,pig")
    s.add_argument("--near", type=float, help="kill: the target closest to this many map units from her (+ = right)")
    s.add_argument("--ground", help="clear: only monsters on these platforms (and any on hers), e.g. 17,18")
    s.add_argument("--kills", type=int, default=6, help="clear: stop after this many kills")
    s.add_argument("--seconds", type=float, default=120, help="stop after this long")
    s.add_argument("--every", type=float, help="look: a look every this many seconds for --seconds, her place in a "
                                               "line each (what looks make of the window moved or resized meanwhile)")
    s.add_argument("--skill", action="store_true", help="throw Lucky Seven instead of the regular attack (MP)")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.add_argument("--potion-hp", type=float, default=0.5, help="drink an HP potion below this HP fraction")
    s.add_argument("--low-hp", type=float, default=0.35, help="stop and hand back below this HP fraction")
    feed_arg(s)
    sweep_arg(s, "off")
    s.set_defaults(func=cmd_hunt)

    s = sub.add_parser("probe", help="in-game probes of what a star hits: list them, run one, score their log")
    probes = s.add_subparsers(dest="action", required=True)
    probes.add_parser("list", help="the probes: where she stands, what she throws at, what each settles "
                                   "(sends no keys)")
    q = probes.add_parser("run", help="run a probe in the game: from her place, throw only at a lone monster in "
                                      "range, and log every throw to lab/probe_geometry.jsonl")
    q.add_argument("name", help="the probe (see `probe list`)")
    q.add_argument("--x", type=float, help="stand at this map x on its platform instead")
    q.add_argument("--dx", type=_span, help="throw at monsters this far off sideways by their middle instead, LO-HI "
                                            "(map units, every kind)")
    q.add_argument("--style", choices=["throw", "jump_throw", "jump_back"],
                   help="throw from the ground, from a jump straight up, or jumping back (default: the probe's)")
    q.add_argument("--throws", type=int, help="how many throws (default: the probe's)")
    q.add_argument("--walls", choices=["on", "off"],
                   help="whether her decisions keep to the wall rule during the run: who counts as in front, and "
                        "fights (default: the probe's; off for those that measure it)")
    q.add_argument("--crowd", type=int,
                   help="also throw with this many other monsters in front (each is logged with what it showed; "
                        "default: the probe's, 2 for claw-edge, else 0)")
    q.add_argument("--seconds", type=float, default=900, help="stop after this long")
    q.add_argument("--map", type=int, default=DEFAULT_MAP)
    q.add_argument("--potion-hp", type=float, default=0.5, help="drink an HP potion below this HP fraction")
    q.add_argument("--low-hp", type=float, default=0.35, help="stop below this HP fraction")
    q.set_defaults(game=True)
    q = probes.add_parser("score", help="settle the probes' questions from their log (sends no keys)")
    q.add_argument("file", nargs="?", help="the log (default: lab/probe_geometry.jsonl)")
    q.add_argument("--map", type=int, help="the map's game data to score it with (default: the log's map)")
    q.add_argument("--line", choices=["level", "v83"],
                   help="score by this line where a question doesn't weigh it (default: reach.LINE, her decisions')")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("map", help="the map's platforms and where she is on them (sends no keys)")
    s.add_argument("--map", type=int, default=DEFAULT_MAP, help="map id (default: Thicket Around the Beach II)")
    s.add_argument("--offline", action="store_true", help="don't look at the game, just draw the map")
    s.set_defaults(func=cmd_map)

    s = sub.add_parser("nav", help="go to a platform: plan a route (ropes, jumps, drops) and follow it")
    s.add_argument("platform", type=int, help="platform id (see `map`)")
    s.add_argument("--x", type=float, help="and walk to this map x on it")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.add_argument("--seconds", type=float, default=180)
    s.add_argument("--fight", action="store_true", help="shoot monsters on the way instead of avoiding them")
    s.add_argument("--avoid", action="append", default=[], metavar="KIND",
                   help="a kind of move (climb, rope_jump, jump, ...) the route takes only if nothing else gets there: "
                        "from 23, `nav 17 --avoid climb` goes by the two rope hops")
    s.add_argument("--ways", type=int, metavar="N",
                   help="each trip chooses among the N best distinct ways by their times (route.ROUTES; default 1: "
                        "the quickest, varied by the moves' random weights; README: Route diversity)")
    feed_arg(s)
    sweep_arg(s, "off")
    s.set_defaults(func=cmd_nav)

    s = sub.add_parser("tour", help="hunt the map: sweep the platforms worth hunting (nearest next, varied a "
                                      "little), clear each, collect the drops, rest when low")
    s.add_argument("--skill", action="store_true", help="throw Lucky Seven instead of the regular attack (MP)")
    s.add_argument("--stops", type=int, default=None,
                   help="how many hunting platforms to visit (default: every one worth it)")
    s.add_argument("--kills", type=int, default=5, help="kills per stop")
    s.add_argument("--seconds", type=float, default=90, help="time per stop")
    s.add_argument("--kinds", help="only these monsters, e.g. slime,pig")
    s.add_argument("--seed", type=int, help="random seed for the order and the route choices (default: random)")
    s.add_argument("--rest-below", type=float, default=0.5, help="go rest when HP is below this between stops")
    s.add_argument("--rest-avoid", action="store_true",
                   help="on the trips to rest and to stand by, keep clear of monsters instead of shooting the ones "
                        "in the way (simulator-tested only)")
    s.add_argument("--potion-hp", type=float, default=0.45, help="drink an HP potion below this while fighting")
    s.add_argument("--low-hp", type=float, default=0.3, help="stop fighting and rest below this")
    s.add_argument("--loop", action="store_true",
                   help="hunt with the hunt loop instead of the sweep's stops (gamelab/docs/hunt-modes.md)")
    s.add_argument("--minutes", type=float, default=20, help="with --loop: how long to hunt, then stand by")
    s.add_argument("--grind", action="store_true",
                   help="with --loop: go round the map in a loop of stops chosen by EXP a minute, shooting what comes "
                        "into reach on the way, instead of hunting one ground at a time (behavior.grind; README: "
                        "Grinding; simulator-tested only)")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.add_argument("--ways", type=int, metavar="N",
                   help="each trip chooses among the N best distinct ways by their times (route.ROUTES; default 1: "
                        "the quickest, varied by the moves' random weights; README: Route diversity)")
    s.add_argument("--ground-tie", type=float, metavar="F",
                   help="with --loop: of the grounds within this share of the best's score, the one she left longest "
                        "ago goes next (hunt.GROUND_TIE; default 0: the best; README: Route diversity)")
    feed_arg(s)
    sweep_arg(s, "on with --loop (hunt.SWEEP), else off")
    s.set_defaults(func=cmd_tour)

    s = sub.add_parser("pet", help="keep her pet fed: its fullness followed in time (read once off her Character Info) "
                                   "and a food when it's low; --every N: every N minutes")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.add_argument("--every", type=float, default=0, help="minutes between follows (0: once; 1 counts only minutes "
                                                            "the game is there)")
    s.add_argument("--read", action="store_true", help="read its fullness off her Character Info now (a double-click)")
    s.set_defaults(func=cmd_pet)

    s = sub.add_parser("watch", help="look at the game every frame with no command running (her place, the monsters "
                                     "followed, deaths and who killed them), for the mini-map; never sends keys or "
                                     "brings the game forward")
    s.add_argument("--hz", type=float, default=30.0, help="frames a second looked at")
    s.add_argument("--minutes", type=float, default=0, help="stop after this long (default: Ctrl+C)")
    s.add_argument("--lie-detector", action="store_true",
                   help="also check the screen for the Lie Detector and page whoever is at the PC")
    s.add_argument("--heading", action="store_true",
                   help="also read which way she faces off every frame and say each change (turn her and watch)")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.set_defaults(func=cmd_watch)

    def map_args(s, web_only: bool) -> None:
        """The mini-map's options (`minimap`, and `web`, which is `minimap --web-only`)."""
        s.add_argument("--replay", help="a telemetry file (lab/telemetry/*.jsonl) or an older sweep track (.json)")
        s.add_argument("--speed", type=float, default=1.0, help="replay speed")
        s.add_argument("--plan", type=int, nargs=2, metavar=("SRC", "DST"), help="show the route between two platforms")
        s.add_argument("--x", type=float, help="--plan: start at this map x on SRC")
        s.add_argument("--map", type=int, default=DEFAULT_MAP)
        if not web_only:
            which = s.add_mutually_exclusive_group()
            which.add_argument("--window-only", action="store_true", help="just the window on this computer")
            which.add_argument("--web-only", action="store_true", help="just the web page (no window)")
            s.add_argument("--width", type=int, default=760, help="the window's width in px")
        s.add_argument("--web-width" if not web_only else "--width", dest="web_width", type=int, default=900,
                       help="the web page's map picture width in px")
        s.add_argument("--port", type=int, default=8765, help="the web page's port")
        s.add_argument("--host", default="0.0.0.0",
                       help="the address the web page is served on: 0.0.0.0 (the default) for a phone on the network, "
                            "127.0.0.1 for this computer only")
        s.set_defaults(func=cmd_minimap)
        if web_only:
            s.set_defaults(web_only=True, window_only=False)

    s = sub.add_parser("minimap", help="the live mini-map: a window beside the game and the same as a web page for a "
                                         "phone (--window-only, --web-only); or --replay a recorded run, --plan a "
                                         "route; sends no keys")
    map_args(s, web_only=False)
    s = sub.add_parser("web", help="the mini-map as a web page only (minimap --web-only); sends no keys")
    map_args(s, web_only=True)

    s = sub.add_parser("sim", help="run a scenario offline on the simulator, recorded for the mini-map (no game)")
    s.add_argument("scenario", help="a scenario name, or `list`")
    s.add_argument("--speed", type=float, default=0,
                   help="also play it in real time, this many times faster (0: as fast as it computes)")
    s.set_defaults(func=cmd_sim)

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

    s = sub.add_parser("sheet", help="contact sheet of a recording: one saved by name, or a folder of JPEG frames "
                                     "with an index (save-recent's, its `before` part, a Lie Detector's), saved in it; "
                                     "--video: the folder as an mp4 instead")
    s.add_argument("recording", help="a recording's name, or a folder")
    s.add_argument("--start", type=float, default=-1e9, help="from this far in: ms (a folder's: s)")
    s.add_argument("--end", type=float, default=1e9, help="to this far in: ms (a folder's: s)")
    s.add_argument("--every", type=int, help="every so many frames (default 3; a folder's 10: a second's)")
    s.add_argument("--cols", type=int, default=8)
    s.add_argument("--shrink", type=int, help="a folder's frames this many times smaller (default 4)")
    s.add_argument("--video", action="store_true", help="a folder: an mp4 playing in real time, its gaps shown")
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

    s = sub.add_parser("bench-look", help="time each stage of a look on saved frames, or --live on the game as "
                                          "it is (refuses unless it's in front; never brings it forward); no keys")
    s.add_argument("files", nargs="*", help="saved screenshots, 1920x1080 or 1366x768, all of one size (default: the "
                                            "1920x1080 ones listed in analysis/bench.py, from gamelab/lab)")
    s.add_argument("--shrink", metavar="WxH", help="shrink the frames to this client size first (1366x768: the same "
                                                   "scenes at the game's own size)")
    s.add_argument("--look-scale", type=int, choices=(1, 2), help="looks at full (1) or half (2) size (default: as "
                                                                  "screen.LOOK_SCALES says for the frames' size)")
    s.add_argument("--reps", type=int, default=5, help="looks at each saved frame")
    s.add_argument("--live", action="store_true", help="look at the game on screen instead (it must be in front)")
    s.add_argument("--n", type=int, default=100, help="--live: how many captures")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.add_argument("--capture", choices=("dxgi", "mss"), help="--live: capture this way (default: config.toml)")
    s.set_defaults(func=cmd_bench_look)
    s = sub.add_parser("bench-capture", help="time a look's capture and a glance's through DXGI and mss on the "
                                             "game as it is on screen; never brings it forward; no keys")
    s.add_argument("--n", type=int, default=100, help="rounds")
    s.set_defaults(func=cmd_bench_capture)
    s = sub.add_parser("bench-paste", help="a map's monsters (those with a colour gate) and the drops drawn onto "
                                           "frames of the game at 1366x768 as the game draws them, looked for at "
                                           "that size and stretched to 1920x1080 (offline; no game, no keys)")
    s.add_argument("files", nargs="*", help="frames of the game at 1366x768 to draw on (default: "
                                            "tests/fixtures/native)")
    s.add_argument("--map", type=int, default=DEFAULT_MAP, help="whose monsters (default: the first map's)")
    s.add_argument("--each", type=int, default=2, help="times each picture is drawn (monsters: each way)")
    s.add_argument("--seed", type=int, default=0, help="where they're drawn")
    s.add_argument("--full-step-1", action="store_true", help="also search with step 1 at full size at 1366x768")
    s.set_defaults(func=cmd_bench_paste)
    s = sub.add_parser("learn-colours", help="a map's background colours from saved frames of it, for the colour "
                                             "gates of its monsters without a hand-made one (offline; no game, no "
                                             "keys)")
    s.add_argument("files", nargs="+", help="saved frames of the map (full-size screenshots of the game window, any "
                                            "size), or folders of them")
    s.add_argument("--map", type=int, required=True, help="the map's id")
    s.add_argument("--rounds", type=int, default=2, help="times the monsters are found and cut out "
                                                         "(sprites.CUT_ROUNDS)")
    s.add_argument("--out", help="the folder to keep it in (default: gamelab/assets/backgrounds)")
    s.set_defaults(func=cmd_learn_colours)
    s = sub.add_parser("bodies", help="where the monster detector puts monsters' feet on saved screenshots, and "
                                      "their x from frame to frame on recordings (offline; no game, no keys)")
    s.add_argument("files", nargs="*", help="saved screenshots, 1920x1080 or 1366x768, all of one size (default: "
                                            "bench-look's, from gamelab/lab)")
    s.add_argument("--recordings", nargs="*", help="recordings (names in gamelab/lab/recordings, or .npz files) "
                                                   f"with the camera still (default: {', '.join(BODY_RECORDINGS)})")
    s.add_argument("--tracks", nargs="+", help="instead: telemetry files of `watch` (.jsonl): each track's velocity "
                                               "against how its x moved")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.set_defaults(func=cmd_bodies)

    s = sub.add_parser("recorder", help="the rolling recording: the game's window 10 times a second, the last 5 "
                                        "minutes kept in memory, saved on a Lie Detector alarm and by save-recent; "
                                        "reads the screen only (no keys, never brings the game forward)")
    s.add_argument("--status", action="store_true", help="how the running recorder is doing (starts none)")
    s.add_argument("--minutes", type=float, default=0, help="stop after this long (default: Ctrl+C)")
    s.set_defaults(func=cmd_recorder)
    s = sub.add_parser("save-recent", help="save the running recorder's last minutes to gamelab/lab/recent/<time>/ "
                                           "(frames, index, summary) and print the folder")
    s.add_argument("--seconds", type=float, default=300.0, help="how far back (default 300, all it keeps)")
    s.add_argument("--name", help="a word added to the folder's name")
    s.set_defaults(func=cmd_save_recent)

    s = sub.add_parser("heading", help="which way she faces on saved frames (her picture above her name tag), and "
                                       "her pictures learnt from frames whose facing you know (offline; no game, no "
                                       "keys)")
    s.add_argument("frames", nargs="*", help="saved frames (PNG: the game's whole client area), all of one size")
    s.add_argument("--learn", choices=(LEFT, RIGHT), help="these frames show her facing this way: her pictures learn "
                                                         "them")
    s.add_argument("--seed", action="store_true", help="the seed in gamelab/assets (in git), which stands in until "
                                                       "she has pictures of her own in gamelab/lab, not those")
    s.add_argument("--fresh", action="store_true", help="drop the pictures first (a new outfit)")
    s.add_argument("--tags", help="where her tag is on the frames, not where a look finds it: a JSON list of "
                                  "{\"file\", \"tag\": [x, y]} (look-frame px), by file name")
    s.set_defaults(func=cmd_heading)

    s = sub.add_parser("lab-usage", help="what gamelab keeps under gamelab/lab (recordings, the Lie Detector's "
                                         "recordings, screenshots, telemetry, harness runs, golden traces), by kind; "
                                         "deletes nothing")
    s.add_argument("--prune-telemetry", action="store_true",
                   help="delete the telemetry a run starting now would (the oldest past 2 GB in all; none for "
                        "its age); recordings and screenshots are never deleted (the Lie Detector's past their "
                        "10 GB, when one starts)")
    s.set_defaults(func=cmd_lab_usage)
    s = sub.add_parser("coverage", help="how often each route move was planned, taken and missed, the moves never "
                                        "taken and why, from harness runs and live telemetry; a picture of the map "
                                        "(offline; no game, no keys)")
    s.add_argument("runs", nargs="*", help="saved harness runs (names in gamelab/lab/harness; default: none, the live "
                                           "telemetry alone)")
    s.add_argument("--live", action="store_true", help="the live telemetry too (gamelab/lab/telemetry, but the "
                                                       "simulator's files)")
    s.add_argument("--text", help="also write the report to this file")
    s.add_argument("--picture", help="the map with the first set's moves drawn, the others' beside it (-2, -3: "
                                     "default gamelab/lab/coverage_<map>.png)")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.add_argument("--ways", type=int, metavar="N",
                   help="how often the runs' plans would go through a move with each trip choosing among N ways "
                        "(route.ROUTES; default: the navigator's own)")
    s.set_defaults(func=cmd_coverage)
    s = sub.add_parser("respawn", help="how long monsters take to come back after a kill, how long drops lie, and "
                                       "how full the platforms are, from telemetry (offline; no game, no keys)")
    s.add_argument("files", nargs="*", help="telemetry files (.jsonl; default: the live ones in gamelab/lab/telemetry, "
                                            "but the simulator's)")
    s.add_argument("--text", help="also write the report to this file")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.set_defaults(func=cmd_respawn)

    s = sub.add_parser("memory", help="her memory of the monsters run again on a recorded run: what it would have held "
                                      "at each look, for the mini-map (offline; no game, no keys)")
    s.add_argument("file", help="a telemetry file (.jsonl)")
    s.add_argument("--out", help="write the run with 'memory' records in it here (for minimap --replay)")
    s.add_argument("--map", type=int, help="the map (default: the run's)")
    s.set_defaults(func=cmd_memory)

    s = sub.add_parser("patrol", help="grinding's view of a map: its stops, the loop and camping's best; with a "
                                      "recorded run, the next stop it would have taken along it (offline; no game, "
                                      "no keys)")
    s.add_argument("file", nargs="?", help="a telemetry file (.jsonl) to run through her memory and the belief")
    s.add_argument("--every", type=float, default=5.0, help="s between the choices printed along the run")
    s.add_argument("--map", type=int, help="the map (default: the run's, else the first)")
    s.set_defaults(func=cmd_patrol)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Real pixels for window positions and screen captures, whatever the display scaling is set to.
    from maplehelper.win32 import enable_dpi_awareness
    enable_dpi_awareness()
    try:
        if args.cmd in GAME_COMMANDS or getattr(args, "game", False):
            client.check_alarm()
        return args.func(args)
    except client.LieDetectorAlarm as e:
        print(f"LIE DETECTOR: {e}. gamelab stopped all input (a running player stays up, locked, recording it; it "
              "takes only her pet's food, and a hunt's retreat to a safe place in the first 12 s).\n"
              "Answer it in the game yourself. Once it has been off the screen for a minute, run "
              "`python -m gamelab clear-alarm`, then start the player again.")
        return LIE_DETECTOR_EXIT
    except (client.PlayerError, RuntimeError) as e:
        print(f"error: {e}")
        return 1
    finally:
        _close_feeds()


if __name__ == "__main__":
    raise SystemExit(main())
