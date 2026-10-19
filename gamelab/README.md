# gamelab

Test MapleHelper's combos in the real game: play exact key timelines, record the game, and measure what happened. That covers whether she jumped, which way she moved, which way she threw, and whether the game ignored a key.

This is how the timing in `config.toml` was found. A regular attack locks out jumps and attacks for about 730 ms. Quick taps right after takeoff were missed about 6% of the time.

## How it works

MapleStory runs as administrator, and Windows blocks key presses from normal programs into admin windows. So one small process, the **player**, runs elevated, and everything else runs normally and talks to it through files in `gamelab/lab/`.

| Piece | Runs as | Does |
|---|---|---|
| `player.py` | admin (one UAC prompt per session) | Reads plans (JSON) from `lab/inbox`. Presses keys only while the game is in front. Records the game window. Writes results to `lab/outbox`. |
| `client.py` | normal | Submits plans, waits for results, and saves recordings to `lab/recordings/` |
| `timeline.py` | normal | Runs MapleHelper's own combo code on a virtual clock, so tests play exactly what the helper plays |
| `track.py`, `camera.py`, `analyze.py` | normal | Find her by her name tag, correct for camera scrolling, detect jumps and claw swings, judge each combo |

What keeps the player safe:
- **No code from plans:** it runs no code it receives; plans are data.
- **Her keys only:** it only presses the keys `config.toml` `[keys]` binds to her actions, plus `\` (key bindings window) and Esc (close windows). Enter is never pressed, so nothing typed in chat can be sent.
- **No clicks:** it can move the mouse inside the game window to hover over things, but it cannot click.
- **Game in front only:** it stops pressing keys if the game leaves the foreground.
- **Fixed at start:** it loads its code and key list once, when it starts.
- **Stops itself:** it exits when `lab/STOP` appears or after 3 idle hours.

## Use

From the repo root, in the `maplehelper` conda env:

```bash
python -m gamelab player --elevate
python -m gamelab calibrate
python -m gamelab combo jump_back_shot --runs 4 --chain 8
python -m gamelab stop
```

1. `player --elevate` starts the player. Approve the UAC prompt.
2. `calibrate` finds her on screen. Stand her on flat, textured ground, such as a girder, with nothing around.
3. `combo` plays the combo, here 4 runs of 8 shots in a row, and reports every jump and throw.
4. `stop` shuts the player down.

Other commands (`python -m gamelab -h` lists them all):

| Command | What it shows |
|---|---|
| `combo NAME --regular-attack` | Same test, throwing with the plain attack key (no MP) |
| `do walk left 150`, `do climb_rope 300`, `do pick_up 2`, `do rest 40` | Run one move from `maplehelper/actions.py` in the game, recorded |
| `act right*800 alt 500 @960,540 --record` | Ad-hoc steps: `KEY` tap, `KEY*MS` hold, `KEY+`/`KEY-` press/release, `MS` wait, `@X,Y` move the mouse |
| `track RECORDING` | Her path through a recording: position, height, world x, jumps |
| `lock attack 700 740 780` | Whether a jump that many ms after an attack goes through |
| `snap` | Screenshot of the game in `lab/snap.png` |
| `sheet RECORDING --around-her --start 0 --end 1500` | Zoomed contact sheet, to check a recording by eye |
| `check RECORDING` | Whether keys went out on time, and whether the game kept a steady frame rate |
| `list` | Saved recordings |

## Limits

- **Recalibrate** after moving the game window, changing maps or changing resolution. Her name-tag template is kept between calibrations; `calibrate --fresh` rebuilds it, and should be run where no rope crosses her name.
- **It sees the screen only after each step,** so it can't react to monsters in real time. Monsters that bump into her knock her off ropes and swallow key presses.
- **Heights are screen-based,** so they read low when the camera scrolls up with her.
- **Camera correction needs ground with a pattern** under her, like a girder or tiles. Plain ground can't be tracked.
- **Swing detection is tuned to the claw's white/lavender swing effect.** Other weapons or skills may need `SWING_MIN_PIXELS` in `track.py` adjusted.
- **Lucky Seven costs about 7 MP per cast.** A 32-shot test uses about 220 MP. Use `--regular-attack` for long runs.
