"""What gamelab keeps under gamelab/lab, by kind, and how much (`python -m gamelab lab-usage`). It only
reads. Recordings and screenshots are measurement evidence and are never deleted for you; telemetry is
pruned when a run starts or goes on in a new part (telemetry.prune: 14 days, 2 GB), and `lab-usage
--prune-telemetry` applies the same now.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

from .. import telemetry
from ..paths import LAB

# The folders of lab/ and what they hold (anything else is "other": the player's log, learned stats such
# as shots.jsonl and grounds.json, templates, notes); the *.png at the top are screenshots.
FOLDERS = {"recordings": "recordings", "telemetry": "telemetry", "harness": "harness runs",
           "golden": "golden traces", "inbox": "player inbox, outbox", "outbox": "player inbox, outbox"}
KINDS = ("recordings", "screenshots", "telemetry", "harness runs", "golden traces", "player inbox, outbox",
         "other")


@dataclass
class Usage:
    """Files of one kind: how many, their bytes, and when the oldest and newest were last written."""
    kind: str
    files: int = 0
    bytes: int = 0
    oldest: float | None = None
    newest: float | None = None

    def add(self, st: os.stat_result) -> None:
        self.files += 1
        self.bytes += st.st_size
        self.oldest = st.st_mtime if self.oldest is None else min(self.oldest, st.st_mtime)
        self.newest = st.st_mtime if self.newest is None else max(self.newest, st.st_mtime)


def usage(lab: Path = LAB) -> list[Usage]:
    """Every file under `lab`, by kind (KINDS, in that order)."""
    out = {kind: Usage(kind) for kind in KINDS}
    for root, _, names in os.walk(lab):
        parts = Path(root).relative_to(lab).parts
        for name in names:
            try:
                st = os.stat(os.path.join(root, name))
            except OSError:  # gone meanwhile
                continue
            if parts:
                kind = FOLDERS.get(parts[0], "other")
            else:
                kind = "screenshots" if name.lower().endswith(".png") else "other"
            out[kind].add(st)
    return list(out.values())


def report(lab: Path = LAB, now: float | None = None) -> list[str]:
    """The lines `lab-usage` prints: each kind's size and file count, the dates they span, and the
    telemetry a run starting now would delete."""
    kinds = usage(lab)
    total = sum(u.bytes for u in kinds)
    lines = [f"{lab}: {size(total)} in {sum(u.files for u in kinds)} files"]
    for u in kinds:
        span = "" if u.oldest is None else f"  {_day(u.oldest)} to {_day(u.newest)}"
        lines.append(f"  {u.kind:22s} {size(u.bytes):>9s} {u.files:6d} files{span}")
    due = telemetry.prune(lab / "telemetry", now=now, dry_run=True)
    lines.append(f"telemetry is kept {telemetry.KEEP_DAYS} days and {size(telemetry.KEEP_BYTES)} at most, in "
                 f"parts of {size(telemetry.MAX_BYTES)}; due to go now: {len(due)} files "
                 f"({size(sum(b for _, b in due))}), deleted when the next run starts or with --prune-telemetry")
    lines.append("recordings and screenshots are never deleted for you")
    return lines


def size(n: float) -> str:
    """Bytes for people: 512 B, 3.4 KB, 1.3 GB."""
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _day(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(t))
