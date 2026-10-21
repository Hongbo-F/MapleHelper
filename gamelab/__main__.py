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

import numpy as np
from PIL import Image

from maplehelper.combos import COMBOS
from maplehelper.engine import LEFT, RIGHT

from . import paths, telemetry, window
from .analysis import analyze, calibration, recordings
from .analysis.track import airborne, track
from .control import client
from .control.timeline import action_events, chain, her_setup
from .paths import LAB, PLAYER_LOG, RECORDINGS, REPO, STOP

DEFAULT_MAP = 10000012  # Victoria Road: Thicket Around the Beach II
FRONT_WAIT_S = 3.0  # s a hunting command waits for the game to come to the front before it gives up (_to_game)
# `watch`: how often it publishes what it sees (telemetry), and says how it's going on the console. Scenes 10
# times a second and tracks twice a second: on the saved frames a scene was 276 bytes and the tracks 845, so
# about 16 MB an hour, 400 MB a day: the 2 GB telemetry keeps (telemetry.KEEP_BYTES) hold 5 days of it.
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
    from .control.alarm import clear_alarm
    info = clear_alarm()
    if info is None:
        print("no Lie Detector alarm is set")
        return 0
    print(f"cleared the Lie Detector alarm from {info.get('time', '?')} (screenshot: {info.get('screenshot', 'none')})")
    print("gamelab can be started again. The game flagged automated play once; keep sessions short and supervised.")
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
    print(f"check {LAB / 'calibrate_crop.png'}: her name tag should be near the middle, a bit below centre")
    return 0


def cmd_snap(args) -> int:
    window.bring_to_front(_game_title())
    frame = recordings.screenshot("snap", scale=args.scale)
    path = LAB / "snap.png"
    Image.fromarray(frame).save(path)
    print(path)
    return 0


def parse_steps(steps: list[str], tap_ms: int = TAP_MS) -> tuple[list, int]:
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


def _tap(key: str, at_ms: int) -> list[list]:
    """Plan events that tap `key` at `at_ms`."""
    return [[at_ms, key, "down"], [at_ms + TAP_MS, key, "up"]]


def cmd_act(args) -> int:
    """Run a few explore steps, then save a screenshot (and optionally a recording around her)."""
    window.bring_to_front(_game_title())
    events, end = parse_steps(args.steps)
    duration = end + args.settle
    if args.record:
        cal = calibration.load()
        rec = recordings.record(events, name=args.name, duration_ms=duration, crop=cal.crop, scale=cal.scale, fps=30)
        path = LAB / f"{args.name}_sheet.png"
        recordings.contact_sheet(rec.frames, rec.times, path, cols=8, every=args.every)
        print(path)
        frame = recordings.screenshot("act", scale=args.scale, duration_ms=100)  # the steps already ran
    else:  # a few full-size frames while the steps run (~9 over the run, 2-10 fps); only the last is kept
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
    cal = calibration.load()
    rec = recordings.record(events, name=args.name, duration_ms=end + 400, crop=cal.crop, scale=cal.scale, fps=30)
    recordings.contact_sheet(rec.frames, rec.times, LAB / f"{args.name}_sheet.png", cols=8, every=args.every)
    path = LAB / f"{args.name}.png"
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
    """Eyes on the game, with monster templates for that map (screenshots + the game's sprites)."""
    from .perception.eyes import Eyes
    from .perception.sprites import detector_for
    return Eyes(title, calibration.load().template, detector_for(model))


_FEEDS: list = []  # continuous feeds started by this command (--feed), closed when it ends


def _hunter(args, **kw):
    """The hunter on map `args.map`, logging to the console (sim.scenarios builds its stack the same way); with
    `--feed`, her looks and glances from a continuous feed (localization.feed: the game looked at every frame
    in a process of its own)."""
    from .behavior.hunter import Hunter
    from .localization import mapdata
    cfg, keys, _ = her_setup()
    model = mapdata.load(args.map)
    feed = _feed(model, cfg.game_window_title) if getattr(args, "feed", False) else None
    return Hunter(_eyes(model, cfg.game_window_title), keys, model, log=_log, feed=feed, **kw)


def _feed(model, title: str):
    """A continuous feed of the game for this command (its child process; closed when the command ends), or
    None if it can't start (the command then looks for itself)."""
    import functools

    from .attack import reach
    from .localization import feed
    try:
        client = feed.FeedClient(functools.partial(feed.screen_world, model.id, title, calibration.load().template,
                                                   (reach.LOOK_UP, reach.LOOK_DOWN)))
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
    while _FEEDS:
        client = _FEEDS.pop()
        bye = client.bye or {}
        client.close()
        print(f"feed: {bye.get('frames', '?')} frames, {bye.get('placed', '?')} placing her, "
              f"{bye.get('band', '?')} band and {bye.get('wide', '?')} whole-screen searches; "
              f"{client.served} looks and glances from it, {client.refused} taken without it; {client.why or ''}")


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
        return _hunt_look(hunter)
    kinds = set(args.kinds.split(",")) if args.kinds else None
    _record(f"hunt-{args.mode}", hunter.model.id)
    if not _to_game():
        return 1
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
    _annotate(final, LAB / "hunt_end.png")
    left = ", ".join(f"{d.item.kind} at {d.x - final.where.x:+.0f}" for d in final.loot) or "none"
    print(f"done: {hunter.report}; now HP {final.hp:.0%}, MP {final.mp:.0%}; drops left on her platform: {left}"
          + ("; her inventory is full" if hunter.looter.inventory_full else ""))
    return 0


def _hunt_look(hunter) -> int:
    """`hunt look`: what she sees now, printed and drawn on the frame (no keys)."""
    from .attack import reach
    from .localization.scene import find_loot
    s = hunter.look()
    if s.where is None:
        print("can't find her minimap dot")
        return 1
    how = "name tag" if s.tag_seen else "minimap dot + camera (name tag hidden)"
    p = s.platform
    print(f"her: map ({s.where.x:.0f}, {s.where.y:.0f}) on platform {p.id if p else None} ({how}); "
          f"HP {s.hp:.0%}, MP {s.mp:.0%}; room: {s.room('left', reach.EDGE_MARGIN):.0f} left, "
          f"{s.room('right', reach.EDGE_MARGIN):.0f} right")
    for t in s.monsters:
        dx, dy = t.x - s.where.x, t.y - s.where.y
        in_reach = ("in reach" if reach.can_hit(p, s.where.x, s.where.y, t, others=s.monsters)
                    else "out of reach" if not reach.can_shoot(dx, dy, kind=t.kind)
                    else "behind a wall" if reach.wall_to(p, s.where.x, s.where.y, t) is not None
                    else "behind one behind a wall")
        facing = {1: ", facing right", -1: ", facing left"}.get(t.facing, "")
        frame = f", {t.mob.template} ({t.mob.frame})" if t.mob.frame else f", {t.mob.template}"
        print(f"  {t.kind} on platform {t.platform}: {dx:+.0f}, {dy:+.0f} ({in_reach}, score {t.mob.score}{facing}{frame})")
    s.loot = find_loot(s, hunter.model, hunter.senses.eyes, wide=True)
    for d in s.loot:
        print(f"  drop: {d.item.kind} on platform {d.platform}: {d.x - s.where.x:+.0f}, "
              f"{d.y - s.where.y:+.0f} (score {d.item.score})")
    _annotate(s, LAB / "hunt_look.png")
    print(LAB / "hunt_look.png")
    return 0


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
            w = Localizer(model).where_dot(s.dot)
            print(f"she's at map ({w.x:.0f}, {w.y:.0f}) on {platform_text(w.platform)} (dot {s.dot[0]:.1f},{s.dot[1]:.1f})")
            marks.append((w.x, w.y, "her"))
    print(f"{model.name}: {len(model.platforms)} platforms, {len(model.ropes)} ropes; monster spawns by platform:",
          dict(sorted(model.spawn_platforms.items())))
    path = LAB / f"map_{args.map}.png"
    mapdata.render(model, path, marks=marks)
    print(path)
    return 0


def cmd_nav(args) -> int:
    hunter = _hunter(args)
    nav = hunter.navigator(fighting=args.fight)
    _record("nav", hunter.model.id)
    if not _to_game():
        return 1
    start = time.monotonic()
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
    hunter = _hunter(args, low_hp=args.low_hp, potion_hp=args.potion_hp)
    nav = hunter.navigator(fighting=True, rng=random.Random(args.seed))  # the route choices too, as the simulator does
    _record("tour", hunter.model.id)
    if not _to_game():
        return 1
    tour = Tour(hunter, nav, log=_log, rng=random.Random(args.seed), rest_below=args.rest_below,
                rest_avoid=args.rest_avoid)
    kinds = set(args.kinds.split(",")) if args.kinds else None
    if args.loop:  # the hunt loop (gamelab/docs/hunt-modes.md): one decision per look, for --minutes
        from .behavior.hunt import Hunt
        if kinds is not None:
            tour.worth = kinds
        report = Hunt(tour, log=_log).run(minutes=args.minutes)
        print(f"hunt done: {report}")
        return 0
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
            server = web.start(feed, host=args.host, port=args.port, key=args.key, ends=ends)
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
    cal = calibration.load()
    window.bring_to_front(cfg.game_window_title)
    for d in args.delays:
        events = _tap(attack.name, 0) + _tap(keys.jump.name, d)
        rec = recordings.record(events, name=f"lock_{args.action}_{d}", duration_ms=d + 900, crop=cal.crop, scale=cal.scale)
        jumped = [s for s in airborne(track(rec.frames, rec.times, cal.template)) if s[0] >= d]
        print(f"jump {d:4d} ms after {args.action}: " +
              (f"airborne at {jumped[0][0]:.0f} ms" if jumped else "IGNORED (still locked)"))
        time.sleep(SETTLE_S)
    return 0


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
        rec = recordings.record(events, name=f"{args.name}_{run}", duration_ms=end + 600, crop=cal.crop, scale=cal.scale)
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


def _around_her(frame: np.ndarray, x: int, y: int) -> np.ndarray:
    """The AROUND_HER box of a frame, around a name tag whose top-left is (x, y). Cut at the frame's
    edges, it is padded with black at its right and bottom."""
    dx, dy, w, h = AROUND_HER
    out = np.zeros((h, w, 3), np.uint8)
    src = frame[max(0, y + dy):y + dy + h, max(0, x + dx):x + dx + w]
    out[:src.shape[0], :src.shape[1]] = src
    return out


def cmd_sheet(args) -> int:
    rec = recordings.load(args.recording)
    keep = [i for i, t in enumerate(rec.times) if args.start <= t <= args.end][::args.every]
    frames, times = rec.frames[keep], rec.times[keep]
    if args.around_her:
        rows = track(frames, times, calibration.load().template)
        frames = np.stack([_around_her(f, x, y) for f, (_, x, y, _) in zip(frames, rows)])
    path = LAB / f"{args.recording}_sheet.png"
    recordings.contact_sheet(frames, times, path, cols=args.cols, every=1,
                             zoom=AROUND_HER_ZOOM if args.around_her else 1)
    print(path)
    return 0


def cmd_track(args) -> int:
    """Her path through a recording: screen position, height above the start, world x, and jumps."""
    from .analysis.scroll import world_x
    rec = recordings.load(args.recording)
    cal = calibration.load()
    rows = track(rec.frames, rec.times, cal.template)
    xs = world_x(rec.frames, rows)
    rest_y = rows[0][2]
    print("   t ms   screen x,y   height   world x   match")
    for i in range(0, len(rows), args.every):
        t, x, y, score = rows[i]
        print(f"  {t:6.0f}   {x:4d},{y:4d}   {rest_y - y:5d}   {xs[i]:7.1f}   {score:.2f}")
    print("airborne:", [(round(a), round(b) if b else None) for a, b in airborne(rows)])
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
    template = calibration.load().template
    frames = None
    if args.live:
        eyes = bench.FrontEyes(title, template, detector_for(model), screen=capture.Grabber(args.capture))
    else:
        files = [Path(f) for f in args.files] or bench.saved_frames()
        if not files:
            print(f"no frames: name some saved 1920x1080 screenshots (the default ones are in {LAB})")
            return 1
        frames = bench.load(files)
        eyes = bench.SavedEyes(template, detector_for(model))
    from .localization.worldstate import Asks, Schedule, WorldStep
    senses = Senses(eyes, model, band=(reach.LOOK_UP, reach.LOOK_DOWN))
    # The continuous feed's step on the same frames (localization.worldstate): her place and every monster it
    # follows re-found, and with what a frame may also be asked for. Each frame is looked at several times in
    # a row, as a still scene; the whole-screen one finds the monsters the others then re-find.
    step = WorldStep(Senses(eyes, model, band=(reach.LOOK_UP, reach.LOOK_DOWN)),
                     schedule=Schedule(band_s=None, wide_s=None))

    def frame_step(asks=None):
        full = eyes._capture()
        return step.step(lambda ropes: eyes.look_at(full, None, ropes), time.monotonic(), True, asks)
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
    print(f"{where}; OpenCV {cv2.__version__}, {cv2.getNumThreads()} threads"
          + ("" if frames is None else "; the grab itself (16.7 ms with mss, about 2 with DXGI) isn't included"))
    print("the C pieces (gamelab/native.c): " + ("in use" if native.lib() is not None
                                                  else f"not in use ({native.why}): numpy instead"))
    print("\n".join(timer.table()))
    if frames is None:
        print(eyes.screen.summary())
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
    detector = detector_for(model)
    files = [Path(f) for f in args.files] or bench.saved_frames()
    eyes = bench.SavedEyes(calibration.load().template, detector)
    senses = Senses(eyes, model, band=(reach.LOOK_UP, reach.LOOK_DOWN))

    def look_at(bgra):
        def look():
            eyes.show(bgra)
            return senses.look(wide=True)
        return look

    scenes = [(name, look_at(bgra)) for name, bgra in bench.load(files)]
    names = BODY_RECORDINGS if args.recordings is None else args.recordings
    found = [Path(r) if Path(r).suffix else RECORDINGS / f"{r}.npz" for r in names]
    print("\n".join(bodies.run(scenes, model, [p for p in found if p.exists()], detector)))
    if missing := [str(p) for p in found if not p.exists()]:
        print(f"recordings not found: {', '.join(missing)}")
    return 0


# --- watching the game, no command running -------------------------------------------------------------

def _watch_runner(args):
    """What `watch` looks with: the game on screen (never brought to the front), her throws unknown (played
    by hand), drops searched on a schedule of their own."""
    from .attack import reach
    from .localization import feed, worldstate
    return feed.screen_world(args.map, _game_title(), calibration.load().template, (reach.LOOK_UP, reach.LOOK_DOWN),
                             throws=False, schedule=worldstate.WATCH)


def cmd_watch(args) -> int:
    """The game looked at every frame with no command running, as `--feed` does for one: her place, the
    monsters tracked (ids, facing, HP bars, deaths and who killed them), her EXP gains and hits, published as
    telemetry for the mini-map. It never sends a key and never brings the game to the front: a frame the
    game isn't wholly showing isn't read. With --lie-detector it also checks the screen for the Lie Detector
    and pages whoever is at the PC (the player keeps its own watchdog regardless)."""
    from .localization import feed, worldstate
    from .perception.lie_detector import LieDetectorSpotter
    feed.quiet()  # below normal priority, as the feed's child runs
    runner = _watch_runner(args)
    model, band = runner.step.model, runner.step.senses.band
    path = telemetry.record_to("watch", telemetry.FOLDER)
    telemetry.emit("session", map=model.id, command="watch", mode="watch")
    print(f"watching the game {args.hz:.0f} times a second; telemetry: {path} (the mini-map follows it); Ctrl+C stops")
    spotter = LieDetectorSpotter() if args.lie_detector else None
    started = time.monotonic()
    until = started + args.minutes * 60 if args.minutes else None
    w = {"scene": -1e9, "tracks": -1e9, "status": started, "lie": -1e9, "visible": None, "loot": [], "frames": 0,
         "seen": 0, "alarm": False}

    def emit(st: worldstate.WorldState, frame) -> None:
        now = time.monotonic()
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
                    _log(f"  {e.data['kind']} #{e.track} died on platform {e.data['platform']}: "
                         + ("hers" if e.data["cause"] == "hers" else "someone else's"))
            if st.loot is not None:
                w["loot"] = st.loot
            if now - w["scene"] >= WATCH_SCENE_S:
                w["scene"] = now
                s = worldstate.to_scene(st, model, band, mobs=True, wide=True)
                s.loot = w["loot"]
                telemetry.emit("scene", **s.record())
            if now - w["tracks"] >= WATCH_TRACKS_S:
                w["tracks"] = now
                telemetry.emit("tracks", list=[tr.record() for tr in st.tracks], frames=st.seq)
            if spotter is not None and frame.full is not None and now - w["lie"] >= WATCH_LIE_DETECTOR_S:
                w["lie"] = now
                if spotter.seen(frame.full) and not w["alarm"]:
                    w["alarm"] = True
                    _lie_detector_seen(frame.full)
        if now - w["status"] >= WATCH_STATUS_S:
            c, rate = runner.step.tracker.counts, w["frames"] / max(1e-9, now - w["status"])
            live = [tr for tr in st.tracks if tr.state != "dead"]
            print(f"{time.strftime('%H:%M:%S')} {rate:.1f} frames a second ({w['seen']} with the game on screen); "
                  f"{len(live)} monsters followed; deaths seen: {c['died hers']} hers, {c['died other']} others'",
                  flush=True)
            w["status"], w["frames"], w["seen"] = now, 0, 0

    def poll(wait: float):
        time.sleep(wait)
        return None, False

    try:
        feed.run(runner, args.hz, poll, emit, until)
    except KeyboardInterrupt:
        pass
    finally:
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


# --- the command line ------------------------------------------------------------------------------------

def feed_arg(s) -> None:
    s.add_argument("--feed", action="store_true",
                   help="looks and glances from a continuous feed: the game looked at every frame in a process of its "
                        "own (not tried in game yet)")


def build_parser() -> argparse.ArgumentParser:
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

    s = sub.add_parser("hunt", help="hunting objectives that run on their own (see gamelab/behavior/hunter.py)")
    s.add_argument("mode", choices=["look", "kill", "clear", "shots"],
                   help="look: what she sees now; kill: one monster; clear: everything in reach, nearest "
                        "first; shots: hits and misses so far, by target height (sends no keys)")
    s.add_argument("--kinds", help="only these monsters, e.g. slime,pig")
    s.add_argument("--near", type=float, help="kill: the target closest to this many map units from her (+ = right)")
    s.add_argument("--ground", help="clear: only monsters on these platforms (and any on hers), e.g. 17,18")
    s.add_argument("--kills", type=int, default=6, help="clear: stop after this many kills")
    s.add_argument("--seconds", type=float, default=120, help="stop after this long")
    s.add_argument("--skill", action="store_true", help="throw Lucky Seven instead of the regular attack (MP)")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.add_argument("--potion-hp", type=float, default=0.5, help="drink an HP potion below this HP fraction")
    s.add_argument("--low-hp", type=float, default=0.35, help="stop and hand back below this HP fraction")
    feed_arg(s)
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
    q.add_argument("--crowd", type=int, default=0,
                   help="also throw with this many other monsters in front (each is logged with what it showed)")
    q.add_argument("--seconds", type=float, default=900, help="stop after this long")
    q.add_argument("--map", type=int, default=DEFAULT_MAP)
    q.add_argument("--potion-hp", type=float, default=0.5, help="drink an HP potion below this HP fraction")
    q.add_argument("--low-hp", type=float, default=0.35, help="stop below this HP fraction")
    q.set_defaults(game=True)
    q = probes.add_parser("score", help="settle the probes' questions from their log (sends no keys)")
    q.add_argument("file", nargs="?", help="the log (default: lab/probe_geometry.jsonl)")
    q.add_argument("--map", type=int, help="the map's game data to score it with (default: the log's map)")
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
    feed_arg(s)
    s.set_defaults(func=cmd_nav)

    s = sub.add_parser("tour", help="hunt the map: sweep the platforms worth hunting (nearest next, varied a "
                                      "little), clear each, collect the drops, rest when low")
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
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    feed_arg(s)
    s.set_defaults(func=cmd_tour)

    s = sub.add_parser("watch", help="look at the game every frame with no command running (her place, the monsters "
                                     "followed, deaths and who killed them), for the mini-map; never sends keys or "
                                     "brings the game forward")
    s.add_argument("--hz", type=float, default=30.0, help="frames a second looked at")
    s.add_argument("--minutes", type=float, default=0, help="stop after this long (default: Ctrl+C)")
    s.add_argument("--lie-detector", action="store_true",
                   help="also check the screen for the Lie Detector and page whoever is at the PC")
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
        s.add_argument("--key", help="the key the web page's address needs (default: a new random one each time; give "
                                     "one to keep a phone bookmark working)")
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

    s = sub.add_parser("bench-look", help="time each stage of a look on saved frames, or --live on the game as "
                                          "it is (refuses unless it's in front; never brings it forward); no keys")
    s.add_argument("files", nargs="*", help="saved 1920x1080 screenshots (default: the ones listed in "
                                            "analysis/bench.py, from gamelab/lab)")
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
    s = sub.add_parser("bodies", help="where the monster detector puts monsters' feet on saved screenshots, and "
                                      "their x from frame to frame on recordings (offline; no game, no keys)")
    s.add_argument("files", nargs="*", help="saved 1920x1080 screenshots (default: bench-look's, from gamelab/lab)")
    s.add_argument("--recordings", nargs="*", help="recordings (names in gamelab/lab/recordings, or .npz files) "
                                                   f"with the camera still (default: {', '.join(BODY_RECORDINGS)})")
    s.add_argument("--tracks", nargs="+", help="instead: telemetry files of `watch` (.jsonl): each track's velocity "
                                               "against how its x moved")
    s.add_argument("--map", type=int, default=DEFAULT_MAP)
    s.set_defaults(func=cmd_bodies)

    s = sub.add_parser("lab-usage", help="what gamelab keeps under gamelab/lab (recordings, screenshots, "
                                         "telemetry, harness runs, golden traces), by kind; deletes nothing")
    s.add_argument("--prune-telemetry", action="store_true",
                   help="delete the telemetry a run starting now would (the oldest past 2 GB in all; none for "
                        "its age); recordings and screenshots are never deleted")
    s.set_defaults(func=cmd_lab_usage)
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
        print(f"LIE DETECTOR: {e}. gamelab stopped all input and the player shut down.\n"
              "Answer it in the game yourself. Afterwards, run `python -m gamelab clear-alarm`.")
        return LIE_DETECTOR_EXIT
    except (client.PlayerError, RuntimeError) as e:
        print(f"error: {e}")
        return 1
    finally:
        _close_feeds()


if __name__ == "__main__":
    raise SystemExit(main())
