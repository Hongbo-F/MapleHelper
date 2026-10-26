"""Where gamelab's files are: the repo, config.toml and gamelab's assets (in git), and its working
files under gamelab/lab (git-ignored)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CONFIG = REPO / "config.toml"
ASSETS = ROOT / "assets"          # screenshot cut-outs and Lie Detector templates (in git)
HEADING_SEED = ASSETS / "heading.npz"  # her pictures that tell which way she faces until she has her own (HEADING),
                                       # per window size (calibrated: heading_1366x768.npz): perception.heading

LAB = ROOT / "lab"
SHOTS = LAB / "shots"            # the pictures commands save (snap, act, hunt look, map, ...)
INBOX = LAB / "inbox"            # plans waiting for the player
OUTBOX = LAB / "outbox"          # player results
STOP = LAB / "STOP"              # create this file to stop the player
ALARM = LAB / "ALARM.json"       # the Lie Detector was seen; nothing runs until it's cleared
PLAYER_LOG = LAB / "player.log"
RECORDINGS = LAB / "recordings"  # saved frames + key events, by name
LIE_RECORDINGS = LAB / "lie_detector"  # each Lie Detector's whole cycle as the player saw it (control.lie_recorder)
RECENT = LAB / "recent"          # the rolling recording's last minutes, saved on demand (control.rolling)
ALARM_CLIPS = LAB / "alarm_clips"  # each Lie Detector alarm, 10 s before it to 60 s after, the rolling recorder's
                                   # own copy (control.rolling)
RECORDER = LAB / "recorder"      # the rolling recorder's inbox, outbox and status (control.rolling)
CALIBRATION = LAB / "calibration.json"
TAG_TEMPLATE = LAB / "tag_template.npy"
TAG_FULL = LAB / "tag_template_full.npy"  # her name tag at full resolution (made on first look)
HEADING = LAB / "heading.npz"    # her pictures that tell which way she faces, learnt as she plays (perception.heading)
HEADING_SHOTS = LAB / "heading"  # crops of her where the frame and her keys disagreed on her facing (behavior.hunter)
SHOT_LOG = LAB / "shots2.jsonl"  # every throw: where she, the target and the rest in front were, and whether it hit
OLD_SHOT_LOG = LAB / "shots.jsonl"  # ... as logged until 2026-10-20: a kill made every throw in flight a hit
PROBE_LOG = LAB / "probe_geometry.jsonl"  # the in-game probes' throws (behavior.probe): every monster in front,
                                           # where at the pick, and what each showed after the star was due
HIT_LOG = LAB / "hits.jsonl"     # hits taken standing still: how far each pushed her
PET = LAB / "pet.json"           # when she last fed her pet (behavior.hunter.feed_pet_if_due)
GROUNDS = LAB / "grounds.json"   # per map and hunting platform: visits, and the EXP they gave
TELEMETRY = LAB / "telemetry"    # what each run saw and planned, for the mini-map (one file per run)
WINDOW_TRIES = LAB / "window_tries.json"  # when Alt+Enter was tried on the client (control.window_size), by any process


def calibrated(path: Path, suffix: str) -> Path:
    """One of the calibration's files (CALIBRATION, TAG_TEMPLATE, TAG_FULL; her pictures HEADING and HEADING_SEED) for a
    game window of another client size than 1920x1080 (screen.Geometry.suffix: "_1366x768"); `suffix` "" is the file
    itself. Her name tag and her picture are drawn at the game's size, so each size has its own."""
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")
