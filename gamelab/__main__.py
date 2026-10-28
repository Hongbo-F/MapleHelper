"""gamelab's command line (`python -m gamelab -h`): the key player, calibration, exploring and measuring
moves with recordings, hunting, navigation and tours in the game, the mini-map, and the simulator.

The top of the stack: it may import every layer. Only what many commands share (analysis, control and
the shared modules) is imported up here; the other layers are imported by the commands that use them.
"""

import argparse
import ctypes
import os
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
from .perception.framecheck import BadLook
from .paths import LAB, PLAYER_LOG, RECORDINGS, REPO, SHOTS, STOP
from .screen import STRETCHED, geometry

DEFAULT_MAP = 10000012  # Victoria Road: Thicket Around the Beach II
FRONT_WAIT_S = 3.0  # s a hunting command waits for the game to come to the front before it gives up (_to_game)
MINIMIZED_POLL_S = 0.5  # s between looks at whether the minimized game was restored (`watch` waits: _game_geometry)
GAME_POLL_S = 5.0       # s between looks for the game's window while a command waits for it (_wait_for_game)
MINIMAP_TRIES = 3       # times the map's minimap picture is tried before a command that acts stops (_minimap_picture),
MINIMAP_RETRY_S = 0.5   # ... this long apart (in case what kept it from loading passes)
# `watch`: how often it says on the console how what the perception service sees is going (the service publishes the
# telemetry itself: localization.service.SCENE_S, TRACKS_S).
WATCH_STATUS_S = 10.0
WATCH_LIE_DETECTOR_S = 0.5  # --lie-detector: s between checks of the screen for it (the player checks 4 a second)
# Commands that bring the game to the front or send it input; refused while the alarm is set. So is any
# subcommand whose parser sets `game` (`probe run`; `probe list` and `probe score` send nothing).
GAME_COMMANDS = {"player", "calibrate", "window", "snap", "act", "do", "lock", "combo", "hunt", "nav", "tour"}
# Commands that attack in the game (`probe` only with `run`): refused for a character whose attacks aren't built yet
# (the magician's, until her profile reaches the attack layer: _refused_as)
ATTACKING = {"lock", "combo", "hunt", "nav", "tour", "probe"}
# Commands that look at the game or act in it. Until her bottom bar is read (the design's S0g: nothing tells a running
# perception service or pet feeder who plays) each plays another character than the thief only when she's named
# (--character, or GAMELAB_CHARACTER): under `character use`'s file or [characters] default alone the thief plays it, and
# says so (_unnamed), and one that attacks is refused (_refused_as). The perception service is the thief's whoever is
# named (it serves every command, the thief's hunts too, and builds her name tag, pictures and Shape once); on saved
# frames (`perceive --frames`, its own pipe, offline) it's anyone's.
IN_GAME = GAME_COMMANDS | {"pet", "perceive", "watch", "recorder", "save-recent", "spot"}
LIE_DETECTOR_EXIT = 3   # exit code after the Lie Detector stopped a command (the safety stop's: behavior.safety.EXIT,
                        # out of darts': behavior.darts.EXIT)

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


def cmd_character(args) -> int:
    """Who plays (gamelab.character): her profiles, the one a command plays and why; `use <id>`: the one played when
    neither --character nor GAMELAB_CHARACTER names one (lab/character.json); `check`: what the game's bottom bar shows,
    which isn't read yet (perception.whoami, to come): nothing is looked at."""
    from maplehelper.config import ConfigError

    from . import character
    try:
        if args.action == "use":
            if not args.id:
                print(f"use which? one of {', '.join(character.known())}")
                return 2
            klass = character.load(args.id).klass  # (one whose table doesn't parse isn't chosen: why is said)
            path = character.choose(args.id)
            print(f"{args.id} plays from now on when nothing names another ({path}; --character and "
                  f"{character.ENV} still come first)")
            if klass != "thief":
                print(f"  until her bottom bar is read (the design's S0g), a command that looks at or acts in the game "
                      f"plays her only when named (`python -m gamelab --character {args.id} ...`): without it such a "
                      f"command plays the thief and says so, and one that attacks refuses; and perceive and pet "
                      f"processes running now keep the character they started as until they start again")
            return 0
        if args.action == "check":
            print("her bottom bar (job above name) isn't read yet (perception.whoami, to come): nothing was looked at; "
                  f"this process plays {character.active_id()}")
            return 2
        which = character.active_id()
        print(f"plays: {which} (by {character.how_chosen()})")
        if which != character.DEFAULT and not character.named():
            print(f"  (offline only: a command in the game plays her only with --character {which}, until her bottom "
                  f"bar is read)")
        for cid in character.known():
            try:
                lines = character.describe(character.load(cid))
            except ConfigError as e:  # (her table: the others play on)
                lines = [f"{cid}: {e}"]
            for i, line in enumerate(lines):
                print(("* " if cid == which and i == 0 else "  ") + line)
    except ConfigError as e:
        print(f"error: {e}")
        return 2
    print("what the game shows: not read (`character check` is to come)")
    return 0


def _play_as(which: str) -> str | None:
    """--character: `which` plays in this process and its children (gamelab.character.ENV), set before any command
    loads motion.physics (her walk and jumps are worked out once, at its import). What's wrong with it, or None."""
    from maplehelper.config import ConfigError

    from . import character
    try:
        known = character.known()
    except ConfigError as e:
        return f"error: {e}"
    if which not in known:
        return f"error: no character {which!r} (--character): known {', '.join(known)}"
    if (character.resolved() is not None or "gamelab.motion.physics" in sys.modules) and character.active().id != which:
        return f"error: her physics was loaded as {character.active().id} already: --character {which} comes too late"
    os.environ[character.ENV] = which
    return None


def _in_game(args) -> bool:
    return (args.cmd in IN_GAME and not _saved_frames(args)) or getattr(args, "game", False)


def _saved_frames(args) -> bool:
    return args.cmd == "perceive" and bool(getattr(args, "frames", None))


def _unnamed(args) -> str | None:
    """A command in the game (IN_GAME) with another character than the thief chosen only by `character use`'s file or
    [characters] default, none named: the thief plays it (ENV set, for its children too) unless it attacks
    (_refused_as says why not). What it says of that, or None."""
    from maplehelper.config import ConfigError

    from . import character
    if not _in_game(args) or character.named() or args.cmd in ATTACKING:
        return None
    try:
        which, how = character.active_id(), character.how_chosen()
    except ConfigError:
        return None  # (said where it's loaded)
    if which == character.DEFAULT:
        return None
    os.environ[character.ENV] = character.DEFAULT
    return (f"{args.cmd} plays the thief: {how} names {which}, but until her bottom bar is read (the design's S0g) a "
            f"command in the game plays her only when named (--character {which})")


def _refused_as(args) -> str | None:
    """A command that attacks in the game, played by a character whose attacks aren't built yet (a magician's: README,
    Characters), or the perception service played by another than the thief (IN_GAME): why it doesn't run, saying
    what chose her. None for the thief, and for everything else."""
    attacks = args.cmd in ATTACKING and (args.cmd != "probe" or getattr(args, "game", False))
    if not attacks and (args.cmd != "perceive" or _saved_frames(args)):
        return None
    from maplehelper.config import ConfigError

    from . import character
    try:
        c, how = character.active(), character.how_chosen()
    except ConfigError as e:
        return f"error: {e}"
    if c.klass == "thief":
        return None
    if not attacks:
        return (f"perceive plays {c.id} ({c.klass}) by {how}: the perception service is the thief's until it learns "
                f"who plays (the design's S0g), as it serves every command, the thief's hunts too: start it with no "
                f"--character (her own commands use its frames)")
    return (f"{args.cmd} plays {c.id} ({c.klass}) by {how}, whose attacks aren't built yet (gamelab README, "
            f"Characters): only the thief hunts so far; `python -m gamelab character` says who plays")


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
    of it. Sends no keys. With --recover, set a client of another size than 1366x768 (or the game drawing off it) back
    with Alt+Enter twice instead (_window_recover)."""
    if args.recover:
        return _window_recover(args.force)
    if args.size is None:
        print("give the size (1366x768 or 1920x1080), or --recover")
        return 1
    width, height = (int(v) for v in args.size.lower().split("x"))
    result = client.submit(fps=10, duration_ms=200, scale=1, lead_ms=0, name="window", window_client=(width, height))
    path = _shot(f"window_{width}x{height}.png")
    Image.fromarray(result["frames"][-1]).save(path)
    print(f"client area now {result['client']['width']}x{result['client']['height']}: {path}")
    return 0


def _window_recover(force: bool) -> int:
    """`window --recover`: one try at setting the client back to 1366x768 with Alt+Enter twice, full screen and back
    (control.window_size.refit, as a hunt does when its looks have nothing to act on for it; not counted in its tries
    an hour), saying the client's size and place before and after, and how long the window took to settle after each
    Alt+Enter. A client that's 1366x768 already, windowed and drawn where it is (by the perception service's newest
    state), is left as it is, unless `force` (to try it). The Lie Detector is looked for before each Alt+Enter on the
    service's picture, as a viewer of it (started if none runs)."""
    from .control import window_size
    from .localization import service
    from .screen import GAME_VIEW
    title = _game_title()
    hwnd = window.find_game(title)
    area = window.client_rect(hwnd)
    full = window.full_screen(hwnd, area)
    print(f"the game's client: {area['width']}x{area['height']} at ({area['left']}, {area['top']}) (desktop px)"
          f"{', full screen' if full else ''}; DPI awareness {window.dpi_awareness()}", flush=True)
    perception = service.ServiceClient("window", title=title, log=_log)
    _PERCEPTION.append(perception)
    perception.ready(PERCEPTION_WAIT_S)
    problem = perception.problem()
    drawn = problem is not None and problem.kind == "drawn"
    if drawn:
        print(f"the perception service says: {problem.why}")
    if window_size.size_of(area) == GAME_VIEW and not full and not drawn and not force:
        print("it's 1366x768 already, drawn where it is: nothing to do (--force switches full screen and back all the "
              "same, to try it)")
        return 0
    out = window_size.refit(title, GAME_VIEW, log=print, spot=_lie_detector_look(perception))
    if out.held:  # (not tried, and not said: minimized, the alarm, or something going on)
        print(f"no Alt+Enter: {out.why}")
    elif out.after is not None:
        area = window.client_rect(window.find_game(title))
        print(f"the game's client now: {area['width']}x{area['height']} at ({area['left']}, {area['top']})"
              f"{', full screen' if out.full else ''}")
    return 0 if out.ok else 1


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


def _eyes(model, title: str, perception=None, restore: bool = True):
    """A command's eyes on the game, with monster templates for that map (screenshots + the game's sprites), for its
    window at the size it is now (its calibration at that size), and the map's minimap picture (_minimap_picture): what
    it reads off the frames the perception service hands it (`perception`, its client: the whole frames too,
    Eyes.full_frames). They take no shots of their own (perception.eyes.NoScreen). `restore`: a minimized game is
    brought back (the commands that act); else it's waited for. Says where the window is, at what DPI awareness, and
    what minimap picture was loaded: what a run that went wrong is told by (2026-10-24 16:46:50)."""
    from .perception.eyes import Eyes
    from .perception.sprites import detector_for
    geo = _game_geometry(title, restore=restore)
    area = window.client_rect(window.find_game(title))
    print(f"the game's client: {area['width']}x{area['height']} at ({area['left']}, {area['top']}) (desktop px); "
          f"DPI awareness {window.dpi_awareness()}", flush=True)
    eyes = Eyes(title, _calibration(geo).template, detector_for(model, geo), geo=geo,
                minimap_picture=_minimap_picture(model))
    if perception is not None:
        eyes.full_frames = perception.full_frame
    return eyes


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


_PERCEPTION: list = []  # this command's clients of the perception service, closed when it ends
PERCEPTION_WAIT_S = 20.0  # s a command waits at its start for the perception service (started by it if none runs: its
                          # first state comes about 2 s after it starts) before it goes on without (its looks stand
                          # still until it comes)


def _perception(model, title: str, name: str, acts: bool, throws: bool = False):
    """This command's client of the perception service (localization.service: the game looked at all the time, by one
    process; started detached if none runs), named `name`, deciding on map `model`: `acts` (it sends keys: the game is
    brought forward for it, and its map is the one looked at), `throws` (it tells her throws: hunting). Closed when the
    command ends."""
    from .localization import service
    perception = service.ServiceClient(name, acts=acts, throws=throws, map_id=model.id, title=title, log=_log)
    _PERCEPTION.append(perception)
    started = time.monotonic()
    if perception.ready(PERCEPTION_WAIT_S):  # (raises RuntimeError if another command holds it on another map)
        w = perception.welcome
        _log(f"perception: the service (pid {w['pid']}, run {w['epoch']}, map {w['map']}) in "
             f"{time.monotonic() - started:.1f} s")
        if w["map"] != model.id:  # (a viewer, while a command that acts holds it on its own map)
            _log(f"perception: it looks at map {w['map']} for another command, not {model.id}: what it places is on "
                 f"{w['map']}")
    else:
        _log(f"perception: {perception.why}; it's started if none runs, and this command stands still until it comes")
    return perception


def _hunter(args, viewer: bool = False, throws: bool = True, fit=None, **kw):
    """The hunter on map `args.map`, logging to the console (sim.scenarios builds its stack the same way): her looks
    and glances from the perception service (_perception), as a command that sends keys and tells her throws, or with
    `viewer` one that does neither (`hunt look`); `throws` False: it sends keys but doesn't hunt (`pet`). A command that
    hunts sets a client of the wrong size right with Alt+Enter twice (control.window_size.SizeFix): at its start if
    the eyes have no calibration at its size (_fit_start: it waits START_WAIT_S at most, where it stopped), and on its
    looks later (Senses.window_fix), the game at its own 1366x768. `fit(perception)`: what else waits for the client
    before the eyes are made (the pet's: PetWait). A command that hunts has the OCR's models loaded at its start, on a
    thread of their own (perception.messages.warm: the message bar's first read comes in a fight). A command that
    hunts (one that takes `--low-hp`: `hunt`, `tour`, `probe run`) on a map whose monsters are too strong for her
    (behavior.safety.TooStrong: her safety stop's reserve at or over her pet's warning) is refused before it hunts,
    unless `--low-hp` is given."""
    from .behavior.hunter import Hunter
    from .localization import mapdata
    from .screen import GAME_VIEW
    cfg, keys, _ = her_setup()
    title = cfg.game_window_title
    model = mapdata.load(args.map)
    perception = _perception(model, title, args.cmd + ("-" + args.mode if hasattr(args, "mode") else ""),
                             acts=not viewer, throws=throws and not viewer)
    fix = _size_fix(title, perception) if throws and not viewer else None
    if fix is not None:
        _fit_start(title, fix)
    if fit is not None:
        fit(perception)
    eyes = _eyes(model, title, perception, restore=not viewer)
    if throws and not viewer:  # (the OCR's models loaded now, not at the message bar's first read in a fight)
        from .perception.messages import warm
        warm()
    hunter = Hunter(eyes, keys, model, log=_log, feed=perception, **kw)
    strong = getattr(hunter, "too_strong", None) if throws and not viewer and hasattr(args, "low_hp") else None
    if strong is not None and (refused := strong()) is not None:
        raise refused  # (the commands that hunt, which take --low-hp: `nav`'s trips have the stop as it is)
    if fix is not None and eyes.geo.client == GAME_VIEW:  # (Alt+Enter sets the game's own size, not the stretched one)
        hunter.senses.window_fix = fix
    return hunter


START_POLL_S = 1.0  # s between looks at the window (no capture) while a command that acts waits at its start for it
# Guess: ... and that long at most, then it ends (the fix's two tries, AGAIN_S apart, end within a minute and a half
# with a game that never takes the way back; then a minute and a half for the user): it waited for ever, and the
# babysitter kept the pet stopped meanwhile (a command that hunts running), unfed.
START_WAIT_S = 180.0


def _calibrated(size: tuple[int, int]) -> bool:
    """Whether the eyes have a calibration for a client of this size (window.CLIENT_SIZES, calibrated)."""
    return size in window.CLIENT_SIZES and calibration.files(geometry(*size))[0].exists()


def _lie_detector_look(perception):
    """control.window_size.refit's look for the Lie Detector before each Alt+Enter: on the perception service's picture
    of the client at whatever size (ServiceClient.full_frame with any_size; it raises if there's none: then no
    Alt+Enter), a window_size.Sighting if it's there (the alarm is raised with it), else None."""
    from .control.window_size import lie_detector_sighting
    return lambda: lie_detector_sighting(perception.full_frame(time.monotonic(), any_size=True))


def _in_play_look(perception):
    """control.window_size.SizeFix's look for the game in play before a try: how well the HUD's buttons match on the
    perception service's picture of the client (framecheck.hud_seen; it raises if there's none)."""
    from .perception.framecheck import hud_seen
    return lambda: hud_seen(perception.full_frame(time.monotonic(), any_size=True))[0]


def _size_fix(title: str, perception):
    """What sets a client of the wrong size (or the game drawing off it) right with Alt+Enter twice, for a command that
    hunts (control.window_size.SizeFix), its looks for the Lie Detector and for the game in play (not at its login
    screen) on the perception service's picture."""
    from .control.window_size import SizeFix
    return SizeFix(title, log=_log, spot=_lie_detector_look(perception), in_play=_in_play_look(perception))


def _fit_start(title: str, fix) -> None:
    """A command that acts, at its start: a client of a size the eyes have no calibration for (full screen at
    3840x2160, say) is set right with Alt+Enter twice at once (`fix`, without its STEADY_S wait), and waited for while
    it isn't (said once; the fix's second try AGAIN_S later, then it's the user's), START_WAIT_S at most: then it ends
    (RuntimeError), as it ended at once with "not calibrated at 3840x2160 yet" before. The Lie Detector alarm stops
    it, as at any such command's start."""
    from .perception.framecheck import size_problem
    from .screen import NATIVE
    said, start = None, time.monotonic()
    while True:
        client.check_alarm()
        geo = _game_geometry(title, restore=True)  # (a minimized game brought back first, as its first look did)
        if _calibrated(geo.client):
            if said is not None:
                _log(f"the game's client is {geo.client[0]}x{geo.client[1]}: going on")
            return
        if fix(size_problem(geo.client, NATIVE), wait=False):
            continue
        if time.monotonic() - start >= START_WAIT_S:
            raise RuntimeError(f"the game's client is still {geo.client[0]}x{geo.client[1]} after "
                               f"{START_WAIT_S:g} s, a size the eyes aren't calibrated for: start it again once it's "
                               f"1366x768")
        if said != geo.client:
            said = geo.client
            _log(f"the game's client is {geo.client[0]}x{geo.client[1]}, a size the eyes aren't calibrated for: "
                 f"waiting {START_WAIT_S:g} s at most for it to be set back (nothing else is sent; Ctrl+C stops)")
        time.sleep(START_POLL_S)


def cmd_keys(args) -> int:
    """The game's KEY BINDINGS window read (perception.keybindings: which key the game has Attack, her regular attack,
    on, the preset in use, what each key does): off saved frames (--frame, nothing sent), or live (--read, the player
    running): the mouse moved off where the window opens, the backslash, a frame read, Esc only with the window seen
    open (with nothing open Esc opens the game's MENU), and a frame that says it closed, as the hunt's check does after a
    doubt (behavior.keycheck.KeyCheck). With --record the backslash's and Esc's plans record the client (RECORD_FPS
    for OPEN_MS and CLOSE_MS) to time the window's opening and closing, saved with a report under
    lab/shots/keybindings-<time>. Nothing else is sent: no attack, no move, no click; with the Lie Detector alarm up,
    nothing."""
    from .perception import keybindings as kb
    reader = kb.Reader()
    if args.frame:
        for path in args.frame:
            frame = np.asarray(Image.open(path).convert("RGB"))
            r = reader.read(frame)
            print(f"{path}: {r.text()}")
            if r.read:
                print(f"  {kb.functions_text(reader.functions(frame, r))}")
        return 0
    if not args.read:
        print("keys: --read (live: opens the KEY BINDINGS window with the backslash, reads it, closes it) or --frame "
              "PNG (a saved frame, nothing sent)")
        return 2
    if client.alarm_up():
        print("the Lie Detector alarm is up: nothing sent")
        return 3
    _record("keys", args.map)
    hunter = _hunter(args, throws=False)
    if not _to_game():
        return 1
    configured = hunter.actions.keys.attack.name
    if args.record:
        return _keys_recorded(hunter, reader, configured)
    from .behavior.keycheck import KeyCheck
    check = KeyCheck(hunter, reader, log=_log)
    r = check._window()
    if check.window_state == "may be open":
        print("THE KEY BINDINGS WINDOW MAY BE OPEN IN THE GAME: close it with Esc or CANCEL")
    if r is None or not r.read:
        print(f"the key bindings: {'no frame' if r is None else r.text()}")
        return 1
    fn = reader.functions(check._frame_read, r) if check._frame_read is not None else {}
    _say_keys(r, fn, configured)
    return 0 if check.window_state == "closed" else 1


RECORD_FPS = 15      # keys --record: frames a second recorded after the backslash and after Esc ...
OPEN_MS = 1500       # ... for this long after the backslash ...
CLOSE_MS = 1200      # ... and after Esc
LATE_S = 1.0         # ... and one frame this long after the backslash's recording if it showed the window not open


def _say_keys(r, fn: dict, configured: str) -> None:
    """A KEY BINDINGS reading said: where the game has Attack against her configured key, and the whole map; a window
    not read, only why (nothing said of where Attack is)."""
    from .perception import keybindings as kb
    print(f"the key bindings: {r.text()}")
    if not r.read:
        print("  where Attack is isn't known: the window wasn't read")
        return
    if any(kb.same_key(k, configured) for k in r.attack):
        print(f"  Attack is on {configured.upper()}, as config.toml [keys] attack has it")
    else:
        print(f"  ATTACK IS NOT ON {configured.upper()} (config.toml [keys] attack): the game has it "
              + (", ".join(k.upper() for k in r.attack) if r.attack else "on no key"))
    if fn:
        print(f"  {kb.functions_text(fn)}")


def _keys_recorded(hunter, reader, configured: str) -> int:
    """keys --read --record: the window opened and closed with each plan recording the client (RECORD_FPS), every
    frame read, the frames kept and a report (lab/shots/keybindings-<time>/report.json): when it showed open after the
    backslash, when closed after Esc. Esc only with it seen open, once more only if it's still seen open."""
    import json
    from .behavior.keycheck import MOUSE_AT
    out = SHOTS / f"keybindings-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"configured": configured}

    def keep(name, frame):
        Image.fromarray(np.asarray(frame)).save(out / f"{name}.png")

    before = hunter.senses.eyes._capture()
    keep("before", before)
    h, w = before.shape[:2]
    report["client"] = [w, h]
    r = reader.read(before)
    if r.open:
        print("the KEY BINDINGS window is open already: read as it is, nothing sent")
        _say_keys(r, reader.functions(before, r) if r.read else {}, configured)
        report["reading"] = r.record()
        (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return 0
    if (w, h) != (1366, 768):
        print(f"the client is {w}x{h}, not 1366x768: nothing sent (the window is read at 1366x768)")
        return 2
    mouse = [0, "mouse", *MOUSE_AT]
    res = client.submit([mouse, [150, "\\", "down"], [190, "\\", "up"]], fps=RECORD_FPS, duration_ms=OPEN_MS,
                        scale=1, lead_ms=0, name="keybindings_open")
    frames, times = res.get("frames") or [], res.get("times")
    reads = [reader.read(np.asarray(f)) for f in frames]
    report["open_frames"] = [[None if times is None else float(times[i]), bool(x.open), round(x.title, 4)]
                             for i, x in enumerate(reads)]
    if not reads or not reads[-1].open:  # (one more frame a second later: its opening isn't measured yet)
        time.sleep(LATE_S)
        late = hunter.senses.eyes._capture()
        keep("late", late)
        r_late = reader.read(late)
        report["late_frame"] = [LATE_S, bool(r_late.open), round(r_late.title, 4)]
        if r_late.open:
            frames, reads = [*frames, late], [*reads, r_late]
    if not reads or not reads[-1].open:
        if frames:
            keep("after_backslash", frames[-1])
        report["result"] = ("the window wasn't seen open (nor in a frame a second later): no Esc sent. If it's open in "
                            "the game, close it with Esc or CANCEL")
        print(report["result"])
        (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return 1
    first = next(i for i, x in enumerate(reads) if x.open)
    keep("open_first", frames[first])
    keep("open", frames[-1])
    last = reads[-1]
    report["reading"] = last.record()
    _say_keys(last, reader.functions(np.asarray(frames[-1]), last) if last.read else {}, configured)
    for go in (1, 2):
        res = client.submit([[0, "esc", "down"], [40, "esc", "up"]], fps=RECORD_FPS, duration_ms=CLOSE_MS, scale=1,
                            lead_ms=0, name=f"keybindings_close{go}")
        frames, times = res.get("frames") or [], res.get("times")
        reads = [reader.read(np.asarray(f)) for f in frames]
        report[f"close{go}_frames"] = [[None if times is None else float(times[i]), bool(x.open)]
                                       for i, x in enumerate(reads)]
        if frames:
            keep(f"closed{go}", frames[-1])
        if not reads:  # (no frame after it: not known, and no Esc sent unseen, the MENU's)
            report["result"] = (f"not known whether it closed (no frames after Esc x{go}): look at the game, close it "
                                "with Esc or CANCEL if it's open")
            break
        if not reads[-1].open:
            report["result"] = f"read and closed (Esc x{go})"
            break
        if len(reads) < 2 or not reads[-2].open:  # (Esc again only with two frames at its end still showing it)
            report["result"] = "not known whether it closed (open in one frame at the end): look at the game"
            break
    else:
        report["result"] = "STILL OPEN after two Esc: close it yourself (Esc or CANCEL)"
    print(report["result"], f"(frames and report: {out})")
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0 if report["result"].startswith("read") else 1


def cmd_pet(args) -> int:
    """Her pet kept fed (behavior.pet): its fullness followed in time and a food given when it's low, read once off her
    Character Info to start from (or now, with --read); with --every, followed every that many minutes until stopped,
    counting only time the game window is there. Keys: pet food; for a read Esc and a double-click on her. With the Lie
    Detector alarm up it goes on: the food by the fullness followed (the player takes it then), no read until the alarm
    is cleared; a food the player refuses or doesn't answer (it isn't running) is tried again at the next follow. Its
    looks and panels are the perception service's frames (it sends keys, but doesn't hunt: the service keeps its idle
    pace for it). With --every and no game running it waits for the game (the babysitter restarted it every 5 s); and
    while the client is another size, not calibrated at its size, or the game draws off it, it waits too, sending
    nothing, its fullness followed (behavior.pet.PetWait: with one Alt+Enter-twice try at most every 30 minutes, once
    the pet is due a food and the game is seen in play), where it ended ("not calibrated at 3840x2160 yet", 23:21 on
    2026-10-25) and was started again every 5 s."""
    from .behavior.pet import WAIT_POLL_S, PetWait
    title = _game_title()
    waits: list = []
    if args.every:
        _wait_for_game(title)

    def fit(perception) -> None:  # (before the eyes: their calibration is for the size the client is)
        wait = PetWait(title, lambda: _pet_unfit(title, perception), log=_log, spot=_lie_detector_look(perception),
                       picture=lambda: perception.full_frame(time.monotonic(), any_size=True))
        waits.append(wait)
        while wait():
            time.sleep(WAIT_POLL_S)

    hunter = _hunter(args, throws=False, fit=fit if args.every else None)
    eyes = hunter.senses.eyes.geo.client if waits else None
    if waits:  # (from here on, the size the eyes are for; Alt+Enter sets the game's own, PetWait's `want`)
        waits[0].unfit = lambda: _pet_unfit(title, hunter.senses.feed, eyes)
    _record("pet", args.map)
    if args.read and client.alarm_up():
        print("pet fullness: not read with the Lie Detector alarm up (the read double-clicks her)")
    elif args.read:
        window.bring_to_front(title)
        print(f"pet fullness: {hunter.pet.check()}")
    if args.read and not args.every:
        return 0
    while True:
        if waits and (size := _pet_resized(title, eyes)) is not None:
            _log(f"pet: the game's client is {size[0]}x{size[1]} now, a size calibrated but not the eyes' "
                 f"({eyes[0]}x{eyes[1]}): the pet ends, for its babysitter to start it again with eyes for it")
            return 0
        if waits and waits[0]():  # (the client unfit: nothing sent, said once)
            time.sleep(WAIT_POLL_S)
            continue
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


def _pet_unfit(title: str, perception, want: tuple[int, int] | None = None):
    """What keeps `pet --every` from following on now (behavior.pet.PetWait), or None: the client full screen, another
    size than `want` (the eyes'; before they're made, any the eyes are calibrated for), the game drawing off it (the
    perception service's newest state), or another process's size fix trying (window_size.trying: `window --recover`
    runs beside the pet, and a food in its full-screen hold had it give up). Read off the window alone, no capture; no
    game window or a minimized one is `tend`'s (its time not counted), not this."""
    from .control import window_size
    from .perception.framecheck import Problem, size_problem
    from .screen import NATIVE
    try:
        hwnd = window.find_game(title)
        if window.minimized(hwnd):
            return None
        area = window.client_rect(hwnd)
    except RuntimeError:
        return None
    if window_size.trying():
        return Problem("window", "another command is setting the game's client right with Alt+Enter")
    size = (area["width"], area["height"])
    if window.full_screen(hwnd, area):
        return Problem("size", f"the game is full screen ({size[0]}x{size[1]})")
    wrong = size != want if want is not None else not _calibrated(size)
    if wrong:
        return size_problem(size, geometry(*want) if want is not None else NATIVE)
    problem = perception.problem() if perception is not None else None
    return problem if problem is not None and problem.kind == "drawn" else None


def _pet_resized(title: str, eyes: tuple[int, int]) -> tuple[int, int] | None:
    """The client's size if it's windowed at a size the eyes are calibrated for that isn't the size of these (`eyes`):
    `pet --every` ends then, for the babysitter to start it again with eyes for that size (waiting, it waited for ever
    with eyes made at 1920x1080 and the game back at its own 1366x768, unfed); None otherwise (a size it waits on)."""
    try:
        hwnd = window.find_game(title)
        if window.minimized(hwnd):
            return None
        area = window.client_rect(hwnd)
    except RuntimeError:
        return None
    size = (area["width"], area["height"])
    if size == eyes or window.full_screen(hwnd, area) or not _calibrated(size):
        return None
    return size


def _close_perception() -> None:
    """Detach the command from the perception service (which goes on), and say what it had of it."""
    while _PERCEPTION:
        perception = _PERCEPTION.pop()
        down = "" if perception.alive else perception.why  # it went down before the command ended
        perception.close()
        lost = getattr(perception, "lost", 0)  # looks whose frame had left the service's ring (frame_of)
        print(f"perception: {perception.served} states handed out, {perception.refused} times none in time, "
              f"{perception.connects} connections, {perception.gaps} states dropped for reading too slowly"
              + (f", {lost} looks' frames gone from the ring (black)" if lost else "")
              + (f"; {down}" if down and down != "closed" else ""))


def _wait_for_game(title: str) -> None:
    """Until the game's window is there (GAME_POLL_S apart), said once: a command that keeps running between sessions
    (`pet --every`) waits for the game instead of ending (the babysitter started it again every 5 s)."""
    said = False
    while True:
        try:
            window.find_game(title)
            return
        except RuntimeError as e:
            if not said:
                print(f"waiting for the game ({e}); nothing is sent meanwhile", flush=True)
                said = True
        time.sleep(GAME_POLL_S)


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
    """Record this run's telemetry (lab/telemetry), for the mini-map to follow live or replay later, Python's garbage
    collections among it once a minute (gcstats)."""
    from . import gcstats
    path = telemetry.record_to(command)
    telemetry.emit("session", map=map_id, command=command, mode="live")
    gcstats.report_every(telemetry.emit, when=telemetry.to_file)
    _log(f"telemetry: {path} (watch it with `python -m gamelab minimap`: a window, and a page for a phone)")


def cmd_hunt(args) -> int:
    if args.mode == "shots":
        for path in ((paths.OLD_SHOT_LOG,) if not paths.of().folder else ()) + (paths.of().shot_log,):  # (hers)
            _shot_summary(path)
        return 0
    if args.mode != "look":
        _record(f"hunt-{args.mode}", args.map)  # (before the hunter, whose start may set the client right: _fit_start)
    hunter = _hunter(args, viewer=args.mode == "look", low_hp=args.low_hp, potion_hp=args.potion_hp,
                     attack=_attack(args))  # (`look`: what her looks make of the game; no keys)
    if args.mode == "look":
        return _hunt_looks(hunter, args.every, args.seconds) if args.every else _hunt_look(hunter)
    kinds = set(args.kinds.split(",")) if args.kinds else None
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
    """`hunt look --every S`: a look every S seconds for `seconds`, her place in a line each (no keys, and the game never
    brought forward): what looks make of the game window moved, covered, minimized or resized while it runs. One that
    fails its checks says why, and after senses.LOST_S of failing it stands still until one passes (Senses.regain), as a
    hunt does (README: Capture checks)."""
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
    """Her EXP number as `hunt look` reads it (perception.exp_number), off the look's box; a look of the perception
    service's state has none (the box is read off a full-size frame: worldstate.to_scene), so then off its full frame
    (Eyes.look on it, no keys). None: no box."""
    from .perception import exp_number
    box = s.exp_box
    if box is None:
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
        path = Path(args.file) if args.file else paths.of().probe_log
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
    _record(f"probe-{p.name}", args.map)  # (before the hunter, whose start may set the client right: _fit_start)
    hunter = _hunter(args, low_hp=args.low_hp, potion_hp=args.potion_hp)  # the regular attack: no --skill
    nav = hunter.navigator()
    if not _to_game():
        return 1
    from .behavior.retreat import on_alarm
    with on_alarm(hunter):  # the Lie Detector seen: she retreats to a safe place, then the command stops
        report = probe.run(hunter, nav, p, seconds=args.seconds, crowd=args.crowd)
    print(f"done: {report}. Logged to {paths.of().probe_log}: `python -m gamelab probe score` reads it")
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
    if not args.offline:  # the perception service's newest state (no keys, the game never brought forward)
        from .localization.worldstate import Asks
        perception = _perception(model, _game_title(), "map", acts=False)
        st = perception.latest(Asks(time.monotonic()), timeout=PERCEPTION_WAIT_S)
        if st is None:
            problem = perception.problem()
            print(f"her place isn't known: {problem.why if problem else 'no state from the perception service'}")
        elif st.dot:
            w = Localizer(model, geo=st.geo).where_dot(st.dot)  # (the dot's px are the frame's window's)
            print(f"she's at map ({w.x:.0f}, {w.y:.0f}) on {platform_text(w.platform)} "
                  f"(dot {st.dot[0]:.1f},{st.dot[1]:.1f})")
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
    _record("nav", args.map)  # (before the hunter, whose start may set the client right: _fit_start)
    hunter = _hunter(args)
    nav = hunter.navigator(fighting=args.fight)
    if args.avoid:  # those kinds of move cost 20 times as much: any other way wins
        weight = nav.weight
        nav.weight = lambda m: weight(m) * (20.0 if m.kind in args.avoid else 1.0)
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
    _record("tour", args.map)  # (before the hunter, whose start may set the client right: _fit_start)
    hunter = _hunter(args, low_hp=args.low_hp, potion_hp=args.potion_hp, attack=_attack(args))
    nav = hunter.navigator(fighting=True, rng=random.Random(args.seed))  # the route choices too, as the simulator does
    if not _to_game():
        return 1
    tour = Tour(hunter, nav, log=_log, rng=random.Random(args.seed))
    kinds = set(args.kinds.split(",")) if args.kinds else None
    from .behavior.retreat import on_alarm
    if args.loop:  # the hunt loop (gamelab/docs/hunt-modes.md): one decision per look, for --minutes
        from .behavior import hunt
        if kinds is not None:
            tour.worth = kinds
        if args.ground_tie is not None:  # grounds about as good: the one she left longest ago (README: Route diversity)
            hunt.GROUND_TIE = args.ground_tie
        with on_alarm(hunter):  # the Lie Detector seen: she retreats to a safe place, then the command stops
            if args.grind:  # going round the map as a player grinds (behavior.grind; README: Grinding; the default)
                from .behavior.grind import Grind
                report = Grind(tour, log=_log).run(minutes=args.minutes)
            else:  # --camp: one ground at a time, waiting there for its respawns
                report = hunt.Hunt(tour, log=_log).run(minutes=args.minutes)
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
    # Senses.look on frames of its own: the step run inline on each when asked (feed.InlineFeed), what's asked searched
    # on it; then the step alone, as the perception service runs it on every frame.
    looks = {"Senses.look: no monsters": lambda: senses.look(mobs=False),
             "Senses.look: her band": lambda: senses.look(),
             "Senses.look: band + loot on her platform": lambda: senses.look(loot=True),
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
                                                  else f"not in use ({native.why}): numpy instead")
          + f"; the monster search's step 1 (gamelab/search.c): {native.search_state()}")
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
    mine = paths.of()  # (hers)
    template, full_path = _calibration(geo).template, paths.calibrated(mine.tag_full, geo.suffix)
    lab, seed = paths.calibrated(mine.heading, geo.suffix), paths.calibrated(mine.heading_seed, geo.suffix)
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

def cmd_watch(args) -> int:
    """What the perception service sees (localization.service: the game looked at every frame at all times, by one
    process, whose telemetry the mini-map follows), said on the console every WATCH_STATUS_S: its frames a second, the
    monsters followed, the deaths it saw and whose, her EXP a minute, who's attached. A viewer of it: it never sends a
    key and never brings the game to the front, starts the service if none runs (service.ensure), and waits for it and
    for the game. With --lie-detector it also checks the screen for the Lie Detector, the service's frame at full size
    every WATCH_LIE_DETECTOR_S, and pages whoever is at the PC (the player keeps its own watchdog regardless); with
    --heading it reads which way she faces off every frame (perception.heading, her pictures in lab or the seed) and
    says each change: turn her with the arrows and watch. (--hz: the service's rate is its own now: service.HZ,
    whoever is attached.)"""
    from .localization import mapdata
    from .perception.drops import unknown_note
    from .perception.lie_detector import LieDetectorSpotter
    title = _game_title()
    model = mapdata.load(args.map)
    if note := unknown_note(model):  # its monsters whose drops aren't known
        print(note)
    perception = _perception(model, title, "watch", acts=False)
    spotter = LieDetectorSpotter() if args.lie_detector else None
    eyes = None
    if args.heading:  # her pictures, read off the service's frames (no shots of its own)
        eyes = _heading_eyes(title)
        bank = eyes.heading_bank()
        print(f"her facing off every frame: {len(bank.refs)} pictures of her from {len(bank.pool)} boxes"
              + ("" if bank.ready else " (none: nothing can be told; `heading --learn` makes some from saved frames)"))
    print("watching what the perception service sees; Ctrl+C stops", flush=True)
    started = time.monotonic()
    until = started + args.minutes * 60 if args.minutes else None
    w = {"status": started, "lie": -1e9, "visible": None, "frames": 0, "seen": 0, "alarm": False, "facing": "",
         "told": 0, "read": 0, "hers": 0, "others": 0, "seq": 0, "exp": None, "run": None}
    try:
        while until is None or time.monotonic() < until:
            perception.wait_frame(0.5)
            now = time.monotonic()
            if perception.run != w["run"]:  # a new run (the map changed, the service started again): numbered from 1
                w["run"], w["seq"] = perception.run, 0
            states = [st for st in list(perception.states) if st.seq > w["seq"] and not st.late]
            for st in states:
                w["seq"], w["frames"] = max(w["seq"], st.seq), w["frames"] + 1
                usable = st.visible and st.bad is None
                if usable != w["visible"]:
                    w["visible"] = usable
                    _log("the game is on screen: watching" if usable else f"not watching: {st.why or st.bad}")
                if not usable:
                    continue
                w["seen"] += 1
                for e in st.events:
                    if e.kind == "died":
                        w["hers" if e.data.get("cause") == "hers" else "others"] += 1
                    elif e.kind in ("exp_gain", "exp_read") and e.data.get("exp") is not None:
                        w["exp"] = e.data["exp"]
                if eyes is not None:
                    frame = perception.frame_of(st)
                    r = eyes.heading(SimpleNamespace(frame=frame, tag=st.tag, tag_seen=st.tag_seen, geo=st.geo, t=st.t))
                    w["read"] += 1
                    w["told"] += r.side is not None
                    said = r.side or f"can't tell ({r.why})"
                    if said != w["facing"]:
                        w["facing"] = said
                        _log(f"  her frame: faces {r.side} (matched {r.score:.2f}, {r.margin:.2f} ahead)" if r.side else
                             f"  her frame: {said}")
            if spotter is not None and not w["alarm"] and now - w["lie"] >= WATCH_LIE_DETECTOR_S:
                w["lie"] = now
                try:
                    full = perception.full_frame(now)
                except BadLook:
                    full = None
                if full is not None and spotter.seen(full):
                    w["alarm"] = True
                    _lie_detector_seen(full)
            if now - w["status"] >= WATCH_STATUS_S:
                rate = w["frames"] / max(1e-9, now - w["status"])
                newest = perception.states[-1] if perception.states else None
                live = [tr for tr in (newest.tracks if newest is not None else []) if tr.state != "dead"]
                status = perception.status() or {}
                names = ", ".join(c["name"] + ("*" if c["acts"] else "") for c in status.get("clients", []))
                told = f"; her facing told on {w['told']} of {w['read']} frames" if args.heading else ""
                exp = "" if w["exp"] is None else f"; EXP {w['exp']:,}"
                print(f"{time.strftime('%H:%M:%S')} {rate:.1f} frames a second ({w['seen']} with the game on screen); "
                      f"{len(live)} monsters followed; deaths seen: {w['hers']} hers, {w['others']} others'{exp}"
                      f"{told}; attached: {names or '?'}", flush=True)
                w["status"], w["frames"], w["seen"], w["told"], w["read"] = now, 0, 0, 0, 0
    except KeyboardInterrupt:
        pass
    print(f"watched {time.monotonic() - started:.0f} s")
    return 0


def _heading_eyes(title: str):
    """Eyes that read which way she faces off the perception service's frames (`watch --heading`: her pictures for the
    window's size; no shots of their own)."""
    from .perception.eyes import Eyes
    geo = _game_geometry(title)
    return Eyes(title, _calibration(geo).template, geo=geo)


def _lie_detector_seen(frame) -> None:
    """The Lie Detector seen while watching: the alarm (gamelab refuses to send keys until it's cleared) and a
    page, once; nothing is pressed."""
    from .control import alarm
    _log("LIE DETECTOR on screen: answer it yourself; gamelab won't send keys until `clear-alarm`")
    if alarm.alarm_info() is None:
        alarm.raise_alarm(frame, "lie_detector (seen by watch)")


# --- the perception service ----------------------------------------------------------------------------

def cmd_perceive(args) -> int:
    """The perception service (localization.service): the game looked at every frame at all times, the latest of what
    it saw handed to every command (its clients) and published for the mini-map (lab/telemetry/<time>-perceive.jsonl).
    It never sends a key and never brings the game forward, and waits for the game window when there's none. One at a
    time on the PC; it ends itself if its frame loop hangs (service.STUCK_S), for the babysitter or a command to start
    another. It keeps the rolling recording (control.recording: its frames 15 times a second, the last 5 minutes,
    what `gamelab recorder` kept) unless --no-record, once no `gamelab recorder` runs. With --frames, saved frames
    instead of the game, round and round, nothing published or recorded (offline), at a pipe, mutex and info file of its
    own (service.SAVED_ADDRESS: the live one starts and runs beside it; a command reaches it with
    GAMELAB_PERCEIVE=saved)."""
    from . import gcstats
    from .attack import reach
    from .localization import feed, service, worldstate
    from .screen import of_frame
    on_saved = bool(args.frames)
    lock = service.only_one(service.SAVED_MUTEX if on_saved else service.MUTEX)  # held until the process ends
    if lock is None:
        print("a perception service " + ("on saved frames " if on_saved else "") + "is running already (one at a time "
              "on the PC)")
        return 0
    service.settle()
    if args.wide_every is not None:
        service.pace(args.wide_every)
        print(f"the whole screen searched every {args.wide_every:g} s (--wide-every; the default "
              f"{worldstate.WIDE_EVERY_S:g})")
    band, title = (reach.LOOK_UP, reach.LOOK_DOWN), _game_title()
    publish, glimpse, recording = None, None, None
    if args.frames:
        files = sorted(str(p) for p in Path(args.frames).glob("*.png"))
        if not files:
            print(f"no frames in {args.frames} (PNG: the game's whole client area, all of one size)")
            return 1
        geo = of_frame(np.asarray(Image.open(files[0]).convert("RGB")))

        def make(map_id, g):
            return feed.saved_world(files, map_id, _calibration(g).template, band, hold=args.hold,
                                    schedule=service.SCHEDULE)

        def where():
            return geo, None
        print(f"perceiving {len(files)} saved frames of {geo.client[0]}x{geo.client[1]}, each {args.hold} in a row, "
              f"at {service.SAVED_ADDRESS} (commands reach it with {service.WHICH}=saved)")
    else:
        def make(map_id, g):
            return service.screen_world(map_id, title, _calibration(g).template, band, geo=g, log=_log)

        def where():
            return service.window_geometry(title)

        def glimpse():  # (the client at any size, for the size fix's and the pet's looks)
            from .perception import frames
            return frames.glimpse(title)
        telemetry.console_in_background()  # (its lines come from the frame loop: a disk that stalls holds up neither)
        path = telemetry.record_to("perceive", telemetry.FOLDER)
        gcstats.report_every(telemetry.emit, when=telemetry.to_file)
        publish = service.Publisher(band, log=_log)
        if not args.no_record:
            from .control.recording import Recording
            recording = Recording(title, log=lambda m: print(m, flush=True))
        print(f"perceiving the game for every command at {service.HZ:g} frames a second; telemetry: {path} (the "
              f"mini-map follows it); " + ("the rolling recording: not kept (--no-record)" if recording is None else
                                           "the rolling recording kept here") + "; Ctrl+C stops", flush=True)
    svc = service.Service(make, args.map, where, address=service.pipe_address(on_saved),
                          info=service.info_path(on_saved), log=_log, publish=publish, stuck_s=service.STUCK_S,
                          glimpse=glimpse, recording=recording)
    try:
        svc.serve(time.monotonic() + args.minutes * 60 if args.minutes else None)
    except KeyboardInterrupt:
        pass
    print(f"perceived: {svc.status()}")
    return 0


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
    from .attack import danger, reach
    from .motion import physics
    from .perception.kinds import kind_of
    records = telemetry.read(Path(args.file)) if args.file else []
    session = next((r for r in records if r.get("topic") == "session"), {})
    model = mapdata.load(args.map or session.get("map") or DEFAULT_MAP)
    exp = {kind_of(n): e for n, e in model.mob_exp.items()}
    best = max(exp.values(), default=0)
    kinds = {k for k, e in exp.items() if e >= 0.25 * best} or None  # tour.WORTH's cut
    planner = Planner(model)
    c = physics.CHARACTER  # the danger hook as grinding's (Hunter.danger: config.toml [character])
    d = danger.of_map(model, danger.Her(c.hp, c.level))
    p = patrol.Patrol(model, kinds, exp, travel=planner.arrive, danger=lambda pid, x: d.at(pid, x))
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
        j, info = p.choose(t, her["platform"], her["x"], current=p.stop_on(her["platform"]), peek=True)
        if j is not None:
            why[info["why"]] += 1
            print(f"{t - start:7.1f}  on {her['platform']} at x {her['x']:.0f}: next {patrol.choice_text(info)}")
    print(f"choices: {dict(why)}; refills x{p.belief.scale:.2f} of the model ({len(p.belief.refills)} watched); others "
          f"take {({k: round(p.belief.take(k), 2) for k in sorted({q.kind for q in p.points})})}")
    return 0


# --- the command line ------------------------------------------------------------------------------------

def attack_arg(s) -> None:
    s.add_argument("--attack", choices=["choice", "skill", "regular"], default="choice",
                   help="which attack a throw is (behavior.attacks): choice (the default), Lucky Seven but the regular "
                        "attack to finish a monster and while her MP is under a cast; skill, Lucky Seven while she has "
                        "the MP for a cast; regular, the regular attack always (out of darts, behavior.darts, is "
                        "told by her Lucky Seven presses: not watched with regular)")
    s.add_argument("--skill", action="store_true", help="the same as --attack skill")


def safety_arg(s) -> None:
    """`--low-hp`: the safety stop's threshold (behavior.safety: 1 or under a share of her max HP, over 1 HP; None: her
    reserve, which the command line doesn't import: the elevated player builds this parser). Until 2026-10-26 `tour`'s
    stopped the fight and sent her to rest on the chair (30%), and `hunt`'s and `probe run`'s handed back at once (35%);
    until 2026-10-28 it was a share always, 65% by default."""
    s.add_argument("--low-hp", type=float, default=None,
                   help="the safety stop's threshold: her HP under it on two looks 0.6 s apart ends the hunt, she goes "
                        "to a safe place and you're paged; 1 or under, a share of her max HP; over 1, HP (default: her "
                        "profile's hits of the map's largest touch, in HP: the thief 2, the magician 3; with an HP "
                        "potion key of her own, five points under --potion-hp if lower; under one touch two looks in a "
                        "row stops it too)")


def _attack(args) -> str:
    """The attack mode `--attack` (or `--skill`) asks for."""
    return "skill" if getattr(args, "skill", False) else getattr(args, "attack", "regular")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gamelab", description="Drive the real game with exact key timelines and measure what happened.")
    p.add_argument("--character", metavar="ID", help="the character this command plays (thief, magician, or a "
                   "config.toml [characters.<id>]'s): before GAMELAB_CHARACTER, `character use` and [characters] "
                   "default; the thief by default")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("player", help="run the key player (use --elevate: the game runs as admin)")
    s.add_argument("--elevate", action="store_true", help="start it as administrator via a UAC prompt")
    s.set_defaults(func=cmd_player)
    sub.add_parser("stop", help="stop the player").set_defaults(func=cmd_stop)
    sub.add_parser("clear-alarm", help="after you've answered a Lie Detector and its recording has ended: let "
                                       "gamelab run again (start the player again after it)"
                   ).set_defaults(func=cmd_clear_alarm)
    s = sub.add_parser("character", help="who plays: her profiles, the one a command plays and why; `use ID` plays ID "
                                         "when nothing names another (lab/character.json); `check`: her bottom bar "
                                         "(not read yet). Offline: no game, no keys")
    s.add_argument("action", nargs="?", choices=["use", "check"])
    s.add_argument("id", nargs="?", help="with `use`: the character (thief, magician, or a [characters.<id>]'s)")
    s.set_defaults(func=cmd_character)
    s = sub.add_parser("spot", help="check the game (or an image) for the Lie Detector; sends no keys")
    s.add_argument("--image", help="check this screenshot instead of the game")
    s.set_defaults(func=cmd_spot)

    s = sub.add_parser("calibrate", help="find her on screen and save her name tag template")
    s.add_argument("--tag", help="name tag centre X,Y in client pixels, if it isn't found automatically")
    s.add_argument("--fresh", action="store_true", help="rebuild the name tag template (stand clear of ropes)")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("window", help="resize the game's client area to 1366x768 (the game's own size: pixel for "
                                      "pixel) or 1920x1080, through the player; sends no keys. --recover: set a "
                                      "client of another size (or the game drawing off it) back to 1366x768 with "
                                      "Alt+Enter twice (full screen and back), as the commands that act do")
    s.add_argument("size", nargs="?", choices=["1366x768", "1920x1080"])
    s.add_argument("--recover", action="store_true", help="Alt+Enter twice, through the player, if the client isn't "
                   "1366x768 or the game draws off it; says the size before and after (the game switches to full "
                   "screen and back: only with someone at the PC)")
    s.add_argument("--force", action="store_true", help="with --recover: switch even if the client is right "
                   "already, to try it")
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
    s.add_argument("args", nargs="*", help="e.g.  walk right 200   climb_rope 300   pick_up 2   stand_up")
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
    attack_arg(s)
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.add_argument("--potion-hp", type=float, default=0.5, help="drink an HP potion below this HP fraction")
    safety_arg(s)
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
    safety_arg(q)
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
    s.set_defaults(func=cmd_nav)

    s = sub.add_parser("tour", help="hunt the map: sweep the platforms worth hunting (nearest next, varied a "
                                      "little), clear each, collect the drops, then stand by on a safe platform")
    attack_arg(s)
    s.add_argument("--stops", type=int, default=None,
                   help="how many hunting platforms to visit (default: every one worth it)")
    s.add_argument("--kills", type=int, default=5, help="kills per stop")
    s.add_argument("--seconds", type=float, default=90, help="time per stop")
    s.add_argument("--kinds", help="only these monsters, e.g. slime,pig")
    s.add_argument("--seed", type=int, help="random seed for the order and the route choices (default: random)")
    s.add_argument("--potion-hp", type=float, default=0.45, help="drink an HP potion below this while fighting")
    safety_arg(s)
    s.add_argument("--loop", action="store_true",
                   help="hunt with the hunt loop instead of the sweep's stops (gamelab/docs/hunt-modes.md)")
    s.add_argument("--minutes", type=float, default=20, help="with --loop: how long to hunt, then stand by")
    # Grinding is the default since 2026-10-27 (the user: "Yes make that the default"; live it made 318-408 EXP a
    # minute on Tree Dungeon on 2026-10-26, camping 283 and 340 on 10-24 and 10-25); --grind is still taken.
    mode = s.add_mutually_exclusive_group()
    mode.add_argument("--grind", dest="grind", action="store_true", default=True,
                      help="with --loop (the default): go round the map in a loop of stops chosen by EXP a minute, "
                           "shooting what comes into reach on the way (behavior.grind; README: Grinding)")
    mode.add_argument("--camp", dest="grind", action="store_false",
                      help="with --loop: hunt one ground at a time instead, waiting there for its respawns and "
                           "leaving by its rate (behavior.hunt; gamelab/docs/hunt-modes.md)")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.add_argument("--ways", type=int, metavar="N",
                   help="each trip chooses among the N best distinct ways by their times (route.ROUTES; default 1: "
                        "the quickest, varied by the moves' random weights; README: Route diversity)")
    s.add_argument("--ground-tie", type=float, metavar="F",
                   help="with --loop --camp: of the grounds within this share of the best's score, the one she left "
                        "longest ago goes next (hunt.GROUND_TIE; default 0: the best; README: Route diversity)")
    s.set_defaults(func=cmd_tour)

    s = sub.add_parser("keys", help="the game's KEY BINDINGS window read: which key Attack (her regular attack) is on; "
                                    "--frame PNG reads a saved frame (nothing sent), --read opens it live with the "
                                    "backslash and closes it with Esc (nothing else sent)")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.add_argument("--read", action="store_true", help="live: open the window (\\), read it, close it (Esc)")
    s.add_argument("--record", action="store_true", help="with --read: record its opening and closing (15 frames a "
                                                          "second) into lab/shots/keybindings-<time>")
    s.add_argument("--frame", nargs="+", help="saved frames (PNG, 1366x768) to read instead: nothing sent")
    s.set_defaults(func=cmd_keys)

    s = sub.add_parser("pet", help="keep her pet fed: its fullness followed in time (read once off her Character Info) "
                                   "and a food when it's low; --every N: every N minutes")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.add_argument("--every", type=float, default=0, help="minutes between follows (0: once; 1 counts only minutes "
                                                            "the game is there)")
    s.add_argument("--read", action="store_true", help="read its fullness off her Character Info now (a double-click)")
    s.set_defaults(func=cmd_pet)

    s = sub.add_parser("perceive", help="the perception service: the game looked at every frame at all times, for "
                                        "every command and the mini-map; one at a time, never sends keys or brings "
                                        "the game forward")
    s.add_argument("--map", type=int, default=DEFAULT_MAP, help="the map it looks at until a command names another")
    s.add_argument("--frames", help="a folder of saved frames (PNG, the game's whole client area, all of one size) "
                                    "looked at instead of the game, round and round (offline)")
    s.add_argument("--hold", type=int, default=30, help="--frames: each looked at this many times in a row")
    s.add_argument("--minutes", type=float, default=0, help="stop after this long (default: Ctrl+C)")
    s.add_argument("--no-record", action="store_true", help="without the rolling recording (`gamelab recorder` keeps "
                                                            "it then)")
    s.add_argument("--wide-every", type=float, metavar="S", help="the whole screen searched every S s without being "
                   "asked (default 1); faster changes what the hunt sees when, not measured on its EXP yet")
    s.set_defaults(func=cmd_perceive)

    s = sub.add_parser("watch", help="look at the game every frame with no command running (her place, the monsters "
                                     "followed, deaths and who killed them), for the mini-map; never sends keys or "
                                     "brings the game forward")
    s.add_argument("--hz", type=float, default=30.0, help="(none now: the perception service's rate is its own)")
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
    # Who plays, before any command loads her physics (gamelab.character); a command that attacks only as the thief.
    if args.character and (refused := _play_as(args.character)):
        print(refused)
        return 2
    if said := _unnamed(args):
        print(said)
    if refused := _refused_as(args):
        print(refused)
        return 2
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
    except Exception as e:
        from .behavior.safety import HuntStop
        if not isinstance(e, HuntStop):  # (the safety stop, out of darts: behavior.safety, behavior.darts)
            raise
        print(f"{e.label}: {e}. {e.after} {e.advice}")
        return e.exit_code
    finally:
        _close_perception()


if __name__ == "__main__":
    raise SystemExit(main())
