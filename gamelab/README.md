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
- **Stops on the Lie Detector:** the player checks the game screen 4 times a second, idle or not. When MapleStory's Lie Detector (its anti-bot check) appears, it:
  1. stops all input within about 10 ms, mid-move included, and releases any held keys;
  2. saves a screenshot and writes `lab/ALARM.json`;
  3. pages whoever is at the PC with alert sounds and a Windows notification that doesn't take focus from the game;
  4. shuts down.

  It never tries to answer the test. Until you run `python -m gamelab clear-alarm`, every command that touches the game refuses to run (exit code 3), including starting the player again. When Claude is driving gamelab, it also pushes the alert to your phone.

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
| `hunt look` | What she sees now: her position, HP/MP, room to walk on her platform, monsters on her floor (annotated `lab/hunt_look.png`); sends no keys |
| `hunt clear --kinds slime,pig --kills 6` | Objective: kill the monsters on her floor, nearest first, looting each kill right away |
| `hunt kill --near -200` | Objective: kill one monster (the one closest to that offset) and loot it |
| `spot`, `spot --image FILE` | Whether the Lie Detector is on screen now (or in a screenshot); sends no keys |
| `clear-alarm` | After you've answered a Lie Detector yourself: lets gamelab run again |
| `lock attack 700 740 780` | Whether a jump that many ms after an attack goes through |
| `snap` | Screenshot of the game in `lab/snap.png` |
| `sheet RECORDING --around-her --start 0 --end 1500` | Zoomed contact sheet, to check a recording by eye |
| `check RECORDING` | Whether keys went out on time, and whether the game kept a steady frame rate |
| `list` | Saved recordings |

## Hunting objectives

`hunt` takes an objective and makes the small decisions itself, looking at the game about 8 times a second. Screen capture needs no admin rights, so only the keys go through the player.

| Piece | Does |
|---|---|
| `mobs.py` | Finds snails, slimes and pigs. It matches each monster's body-colour silhouette first, then checks the best spots with masked colour templates (`assets/mobs/`, cut from screenshots). About 350 ms per look. |
| `hunt.py` | `Eyes`: finds her (name tag at full resolution), the monsters on her floor, the edges of her platform, HP/MP from the bar colours, and camera scroll. `Hunter`: the objectives. |

What an objective does on its own:
- **Range:** keeps the target 110–330 px away, faces it, and throws Lucky Seven.
- **Spacing:** steps away from any monster within 90 px, unless that walks into another one or off her platform.
- **Edges:** never walks off her platform, and skips targets she can't reach from it.
- **Loot:** sweeps each kill's drops right away, pressing pick-up while walking over them. Drops with a monster on them are queued for after a later kill.
- **Health:** drinks an HP potion below `--potion-hp`, and hands control back below `--low-hp`.

Kills are counted when the target stays unseen for about 1.2 s. Checked against EXP, this is close to exact.

## Getting around a map

Navigation is split into modules, each built on the game's own map data (read with `gamedata`, kept off git):

| Module | Does |
|---|---|
| `mapdata.py` | **Map model.** Platforms (footholds joined end to end), walls, ropes and ladders, portals, monster spawns and the minimap's scale. Everything is in map units; the screen shows them 1.40625× larger at 1920×1080. `python -m gamelab map` draws it with her position. |
| `localize.py` | **Where she is.** Her minimap dot converts to map coordinates: one minimap pixel is 16/1.40625 = 11.4 units. The dot is snapped to the platform she's standing on. |
| `route.py` | **Moves and routes.** Builds every move the physics allows: rope climbs and climb-downs, jumping up through a platform overhead, running jumps (from `jumpSpeed` 555 and gravity 2000: 77 units high, 555 ms in the air, at her measured 149 units/s), walking off an end, and dropping through (↓ + jump). Then finds the quickest route with Dijkstra. |
| `moves.py` | **Doing a move.** Closed loop: look, act, look again. Her position along a platform is refined on screen from her distance to its visible end, which is good to a couple of units for takeoffs near edges. Ropes are lined up with the rope detected on screen. |
| `nav.py` | **Navigator.** Plans, does one move, checks which platform she landed on, and replans from there when it isn't the planned one. |
| `tour.py` | **Hunting tour.** Visits spawn platforms in random order and runs `hunt` on each. It only visits platforms where every spawn is a monster `hunt` can recognize. When HP is low, it rests on the chair at the nearest platform with no spawns (monsters never leave their own platform), and it ends standing by there. |

In game, on Thicket Around the Beach II:
- **Navigation:** 21 moves in a row across every move type without a miss, and a 17-move climb from the basin to the top. The fixes that made that work are in `route.py`: the run-up, simulated landings, and the measured walking model.
- **Tour:** visited the basin, platform 7 and platform 21 in random order, made 5 kills (EXP +64, matching the count), rested from 81% to 97% HP on a safe platform, and stood by.

```bash
python -m gamelab map
python -m gamelab nav 35
python -m gamelab tour --stops 3 --kills 5
```

1. `map` shows her platform and draws the map to `lab/map_10000012.png`.
2. `nav` goes to platform 35 (the right cliff top).
3. `tour` visits 3 hunting platforms in random order, then rests and stands by.

## Limits

- **Recalibrate** after moving the game window, changing maps or changing resolution. Her name-tag template is kept between calibrations; `calibrate --fresh` rebuilds it, and should be run where no rope crosses her name.
- **It sees the screen only after each step,** so it can't react to monsters in real time. Monsters that bump into her knock her off ropes and swallow key presses.
- **Heights are screen-based,** so they read low when the camera scrolls up with her.
- **`hunt` needs a 1920×1080 game window,** the size its templates were made at. It knows only snails, slimes and pigs so far. It doesn't navigate between floors (ropes, ledges): that's still the decider's job, with `do` and `act`.
- **Camera correction needs ground with a pattern** under her, like a girder or tiles. Plain ground can't be tracked.
- **Swing detection is tuned to the claw's white/lavender swing effect.** Other weapons or skills may need `SWING_MIN_PIXELS` in `track.py` adjusted.
- **Lucky Seven costs about 7 MP per cast.** A 32-shot test uses about 220 MP. Use `--regular-attack` for long runs.
