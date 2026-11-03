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
GPU = LAB / "GPU"                # create this file for the re-finds' matches on the GPU (native.GPU; GAMELAB_GPU wins)
ALARM = LAB / "ALARM.json"       # the Lie Detector was seen; nothing runs until it's cleared
ALARM_KEYS = LAB / "alarm_keys.json"  # her moves' and pet food's keys as the running commands follow them: the player's
                                      # allowance with the alarm up (control.alarm.AlarmKeys)
PLAYER_LOG = LAB / "player.log"
PLAYER_INFO = LAB / "player.json"  # what the running player can do ({"splice": 1, "pid", "started"}: written as it starts,
                                   # removed as it stops; control.client.can_splice)
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
TOUCHES = LAB / "touches.jsonl"  # her touches as her HP number measured them, exactly (behavior.safety: the safety
                                # stop's touch sizes)
STATS = LAB / "her_stats.json"  # her DEF as her stat window read it or her touches fit it, and her max HP and level as
                                # last read (behavior.bootstrap: perception's, not config.toml's)
STAT_WINDOW = LAB / "stat_window.json"  # what a live capture of her stat window found (perception.stat_window.
                                         # Calibration: whose pictures, where each panel opened): a hunt opens it only
                                         # with this (behavior.statcheck); the client's, not a character's
STAT_TEMPLATES = LAB / "stat_window"    # ... and the panels cut from the live frame where the game data's didn't match
PET = LAB / "pet.json"           # when she last fed her pet (behavior.hunter.feed_pet_if_due)
GROUNDS = LAB / "grounds.json"   # per map and hunting platform: visits, and the EXP they gave
DAMAGE = LAB / "damage.json"     # her attacks' damage as the monsters' bars showed it, learned as she hunts
                                 # (behavior.damage: the reads and each attack's fit)
MOTION = LAB / "her_motion.json"  # her walk and climb as perception's place track measured them, calibrated as she
                                  # hunts (motion.calibrate: each stretch, with its map, when and her SPEED then)
TELEMETRY = LAB / "telemetry"    # what each run saw and planned, for the mini-map (one file per run)
KEYBINDINGS = "keybindings-{}.json"  # (under LAB) a character's key bindings as the game's window last showed them
                                    # (control.keymap: read by behavior.keycheck, `gamelab keys --read`)
WINDOW_TRIES = LAB / "window_tries.json"  # when Alt+Enter was tried on the client (control.window_size), by any process


# Her own files (`of`): each character's name tag, pictures, pet, grounds and logs. The thief's are the names above,
# as they always were; another character's the same names under lab/characters/<her files>/ (gamelab.character's
# Perception.files: "magician"), her heading seed assets/heading_<her files>.npz.
CHARACTERS = "characters"
MINE = {"calibration": "CALIBRATION", "tag_template": "TAG_TEMPLATE", "tag_full": "TAG_FULL", "heading": "HEADING",
        "heading_seed": "HEADING_SEED", "heading_shots": "HEADING_SHOTS", "shot_log": "SHOT_LOG",
        "probe_log": "PROBE_LOG", "hit_log": "HIT_LOG", "touches": "TOUCHES", "stats": "STATS", "pet": "PET",
        "grounds": "GROUNDS", "damage": "DAMAGE", "motion": "MOTION"}


class Files:
    """A character's own files (`of`, MINE's names in lower case): `folder` "" the thief's, this module's names as they
    are when asked (a test puts its own in their place); another's under lab/characters/<folder>/."""

    def __init__(self, folder: str = ""):
        self.folder = folder

    def __getattr__(self, name: str) -> Path:
        if name not in MINE:
            raise AttributeError(name)
        path = globals()[MINE[name]]
        if not self.folder:
            return path
        if name == "heading_seed":
            return path.with_name(f"{path.stem}_{self.folder}{path.suffix}")
        return globals()["LAB"] / CHARACTERS / self.folder / path.name

    @property
    def lab(self) -> Path:
        """Her own folder: lab/ for the thief; another's lab/characters/<folder>/, made by what saves there first
        (calibrate, as L0 has it first: analysis.calibration)."""
        return globals()["LAB"] if not self.folder else globals()["LAB"] / CHARACTERS / self.folder

    def __repr__(self) -> str:
        return f"Files({self.folder!r})"


def of(character=None) -> Files:
    """Her own files: `character`'s (a gamelab.character.Character; None: the one this process plays)."""
    if character is None:
        from . import character as profiles  # (a module beside this one: loaded when first asked)
        character = profiles.active()
    return Files(character.perception.files)


def calibrated(path: Path, suffix: str) -> Path:
    """One of the calibration's files (CALIBRATION, TAG_TEMPLATE, TAG_FULL; her pictures HEADING and HEADING_SEED) for a
    game window of another client size than 1920x1080 (screen.Geometry.suffix: "_1366x768"); `suffix` "" is the file
    itself. Her name tag and her picture are drawn at the game's size, so each size has its own."""
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")
