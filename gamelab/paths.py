"""Where gamelab's files are: the repo, config.toml and gamelab's assets (in git), and its working
files under gamelab/lab (git-ignored)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CONFIG = REPO / "config.toml"
ASSETS = ROOT / "assets"          # screenshot cut-outs and Lie Detector templates (in git)

LAB = ROOT / "lab"
INBOX = LAB / "inbox"            # plans waiting for the player
OUTBOX = LAB / "outbox"          # player results
STOP = LAB / "STOP"              # create this file to stop the player
ALARM = LAB / "ALARM.json"       # the Lie Detector was seen; nothing runs until it's cleared
PLAYER_LOG = LAB / "player.log"
RECORDINGS = LAB / "recordings"  # saved frames + key events, by name
CALIBRATION = LAB / "calibration.json"
TAG_TEMPLATE = LAB / "tag_template.npy"
TAG_FULL = LAB / "tag_template_full.npy"  # her name tag at full resolution (made on first look)
SHOT_LOG = LAB / "shots2.jsonl"  # every throw: where she, the target and the rest in front were, and whether it hit
OLD_SHOT_LOG = LAB / "shots.jsonl"  # ... as logged until 2026-10-20: a kill made every throw in flight a hit
PROBE_LOG = LAB / "probe_geometry.jsonl"  # the in-game probes' throws (behavior.probe): every monster in front,
                                           # where at the pick, and what each showed after the star was due
HIT_LOG = LAB / "hits.jsonl"     # hits taken standing still: how far each pushed her
GROUNDS = LAB / "grounds.json"   # per map and hunting platform: visits, and the EXP they gave
TELEMETRY = LAB / "telemetry"    # what each run saw and planned, for the mini-map (one file per run)
