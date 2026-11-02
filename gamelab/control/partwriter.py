"""A part of the rolling recording written in a process of its own (control.rolling.write_part_apart, for the
perception service's saves: an alarm's `before/` and clip, `save-recent`), so that the frames' decoding and JPEG or
PNG encoding share no Python lock with the service's frame loop. Reads from stdin, pickled: a header (`folder`,
`info`, `fps`, `png`, `encoders`, `n`), then `n` moments (rolling.Item's fields in order); writes the part
(rolling.write_part) and prints one line of JSON: {"summary": ...}, or {"error": ...} (exit 1). Started as `python -m
gamelab.control.partwriter`. Its threads at the lowest priority (LOWEST) in its process's
below-normal class (Idle, its parent's being Idle): below the frame loop and the game, which come first; as an Idle
background job's at the busiest. Its writers at normal priority (an Idle replay's child: the frame loop's own
priority) still took CPUs from the frame loop on the replays (its fps 11.4 -> 8.0 during a 120 s save, the machine 54%
busy: D:/MapleHelper-scratch/pconst/finish/replay, d_apart2).

Imports: control.rolling.
"""

import json
import pickle
import sys
from pathlib import Path

LOWEST = -2  # THREAD_PRIORITY_LOWEST: this process's threads, its writers too


def main() -> int:
    try:
        import cv2

        from .rolling import Item, low_priority, write_part
        cv2.setNumThreads(1)  # (as `gamelab recorder` ran, rolling.quiet: its writers are the threads)
        low_priority(LOWEST)
        stdin = sys.stdin.buffer
        head = pickle.load(stdin)
        items = []
        for _ in range(int(head["n"])):
            t, data, why, size, key, rgb, seq, run, covered, over = pickle.load(stdin)
            items.append(Item(t, data, why, None if size is None else tuple(size), key, rgb, seq, run, covered,
                              tuple(tuple(b) for b in over)))
        summary = write_part(Path(head["folder"]), items, head["info"], head["fps"], head["png"], head["encoders"],
                             priority=LOWEST)
    except Exception as e:  # (said to the service, which writes the part itself then)
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
        return 1
    print(json.dumps({"summary": summary}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
