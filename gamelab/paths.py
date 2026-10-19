"""Where gamelab keeps its working files (git-ignored)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CONFIG = REPO / "config.toml"

LAB = ROOT / "lab"
INBOX = LAB / "inbox"            # plans waiting for the player
OUTBOX = LAB / "outbox"          # player results
STOP = LAB / "STOP"              # create this file to stop the player
ALARM = LAB / "ALARM.json"       # the Lie Detector was seen; nothing runs until it's cleared
PLAYER_LOG = LAB / "player.log"
RECORDINGS = LAB / "recordings"  # saved frames + key events, by name
CALIBRATION = LAB / "calibration.json"
TAG_TEMPLATE = LAB / "tag_template.npy"
