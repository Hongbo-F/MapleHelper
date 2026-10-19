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
- **Stops on the Lie Detector:** the player checks the game screen 4 times a second, idle or not. It recognizes the countdown window (yellow "LIE DETECTOR" title), the mouse mini-game ("The Lie Detector Mini-game: use the mouse to follow the fading figure") and the Truth Booth that a failed test sends her to. When one appears, it:
  1. stops all input within about 10 ms, mid-move included, and releases any held keys;
  2. saves a screenshot and writes `lab/ALARM.json`;
  3. pages whoever is at the PC with alert sounds and a Windows notification that doesn't take focus from the game;
  4. shuts down.

  It never tries to answer the test. Until you run `python -m gamelab clear-alarm`, every command that touches the game refuses to run (exit code 3), including starting the player again. When Claude is driving gamelab, it also pushes the alert to your phone.

  On 2026-10-18 the mini-game form wasn't known yet. It went unanswered, and she was sent to the Truth Booth (entry count 1). It has been recognized since then.

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
| `hunt look` | What she sees now: her map position, HP/MP, room to walk on her platform, and the monsters in her shooting band with their platform and whether a star reaches them (annotated `lab/hunt_look.png`); sends no keys |
| `hunt clear --ground 17 --kills 6` | Objective: shoot the monsters on platform 17 (and any on hers), nearest in reach first, looting kills that fall on her platform right away |
| `hunt kill --near -200` | Objective: kill one monster (the one closest to that many map units away) and loot it |
| `hunt shots` | Hits and misses of every throw so far, by how far above or below her the target was |
| `spot`, `spot --image FILE` | Whether the Lie Detector is on screen now (or in a screenshot); sends no keys |
| `clear-alarm` | After you've answered a Lie Detector yourself: lets gamelab run again |
| `lock attack 700 740 780` | Whether a jump that many ms after an attack goes through |
| `snap` | Screenshot of the game in `lab/snap.png` |
| `sheet RECORDING --around-her --start 0 --end 1500` | Zoomed contact sheet, to check a recording by eye |
| `check RECORDING` | Whether keys went out on time, and whether the game kept a steady frame rate |
| `list` | Saved recordings |

## Hunting objectives

`hunt` takes an objective and makes the small decisions itself. Screen capture needs no admin rights, so only the keys go through the player.

| Piece | Does |
|---|---|
| `mobs.py` | Finds snails, slimes, pigs and Orange Mushrooms. It matches each monster's body-colour silhouette first, then checks the best spots with masked colour templates. About 300 ms per look. |
| `sprites.py` | Templates for the monsters on the current map, made from the game's own sprites (every stand, move and hit frame) at their on-screen size. They're added to the screenshot cut-outs in `assets/mobs/`, which still catch monsters half hidden in plants. Built on first use and kept with the other extracted data (git-ignored). |
| `shots.py` | Where a star reaches, and where to stand to shoot a platform: the spots on other platforms that have the most of it in reach, safe ones first. Also reads a monster's HP bar. |
| `hunt.py` | `Eyes`: her name tag (full resolution), HP/MP from the bar colours, her minimap dot, and the monsters in her shooting band. `Hunter`: places her and the monsters on the map, and runs the objectives. |

Her position comes from the minimap dot, which is always visible. Her place on screen comes from her name tag. When the tag is hidden, for example behind the party HP box at the map's edge, it comes from the dot and the camera instead. The camera keeps her at a fixed spot on screen except near the map's edges, where it stops, and it learns that spot from the looks where the tag is visible.

What an objective does on its own:
- **Regular attack:** throws with the plain attack key. Lucky Seven costs MP and isn't needed on these monsters (`--skill` uses it anyway).
- **Range:** a regular star flies 400 map units (560 px) and hit a pig 388 units away. She throws as soon as the target is within 340 units, leaving room for it to walk during the half-second wind-up, and only walks closer when it's farther than that. Closer than 90 units, the claw swipes instead of throwing, so she backs off first.
- **Other platforms:** stars pass through platforms, so targets on a platform up to 60 units below or above her are shot too (both measured: a pig 60 below hit, a pig 60 above killed). They can't walk over to her.
- **Placing monsters:** a monster seen on screen goes on the platform its kind spawns on. Feet read from the screen can be 20 units off, which once put pigs "on" the safe platform above them, and she backed away from them and off it.
- **Spacing:** steps away from any monster on her own platform within 110 units, unless that walks into another one or off her platform. A monster between her and where she needs to stand is shot first. Cornered with one too close to throw at (where the claw would only swipe), she jumps over it.
- **Knockback:** a monster touching her pushes her away from it. The hit log (`lab/hits.jsonl`) records each push while no key was pressed: 20-21 units on a wide platform, and 82 and 109 units on a narrow one, which knocked her off. With monsters on her platform she stays 50 units from its ends, and a fight that ends on another platform stops the objective, so the tour sends her back.
- **Facing:** she only taps an arrow to turn when she isn't already facing the target. Every tap also walks her a few units, and nine taps in a row once walked her off the end of her platform.
- **Edges:** never walks off her platform, and skips targets she can't reach from it.
- **Loot:** sweeps each kill's drops right away when they fell on her platform. A sweep never walks past her platform's end: a minimum hold used to walk her off one. Drops elsewhere are remembered for the tour to collect. Drops with a monster on them wait until after a later kill.
- **Health:** drinks an HP potion below `--potion-hp`, and hands control back below `--low-hp`.
- **Shot log:** every throw is logged to `lab/shots.jsonl` with where the target was and whether it was hit. A hit shows as the target's HP bar appearing or dropping. `hunt shots` sums the log up by target height.

A kill counts only when her EXP number in the bottom bar goes up after she started throwing at the target. That number sits on the bar's plain background and doesn't change otherwise, so any change means EXP. Counting a target as dead because it vanished was wrong about half the time, so she used to stop after a hit or two. Hit monsters flinch and hide behind damage numbers, and other players kill monsters too. A target that vanishes without an EXP gain is searched for over a wider area for 3 s, and fought again if it shows up. Otherwise it's counted as lost, not killed, and she gets no loot from it. Next targets are chosen with damaged monsters first, those already showing an HP bar, so survivors get finished.

Checked against her EXP in game: 4 kills gave +52 (Orange Mushroom 15, pig 13, two slimes at 12 each), and 6 kills gave +81. Both match exactly.

## Getting around a map

Navigation is split into modules, each built on the game's own map data (read with `gamedata`, kept off git):

| Module | Does |
|---|---|
| `mapdata.py` | **Map model.** Platforms (footholds joined end to end), walls, ropes and ladders, portals, monster spawns and the minimap's scale. Everything is in map units; the screen shows them 1.40625× larger at 1920×1080. `python -m gamelab map` draws it with her position. |
| `localize.py` | **Where she is.** Her minimap dot converts to map coordinates: one minimap pixel is 16/1.40625 = 11.4 units. The dot is snapped to the platform she's standing on. |
| `route.py` | **Moves and routes.** Builds every move the physics allows: rope climbs and climb-downs, jumping up through a platform overhead, running jumps (from `jumpSpeed` 555 and gravity 2000: 77 units high, 555 ms in the air, at her measured 149 units/s), walking off an end, and dropping through (↓ + jump). Then finds the quickest route with Dijkstra. |
| `moves.py` | **Doing a move.** One look, one key plan, one look to check, the way a player moves. Running jumps run from wherever she is and jump on the way, with no stop at a run-up spot. Ropes are grabbed on the run: she runs at the rope, jumps about 41 units before it so she's at the top of the jump over it, and holds up. Walks and drops go in one plan. Only climbing down still lines up on screen first. Her position along a platform is refined from ropes or the platform's visible end when her name tag is visible. |
| `nav.py` | **Navigator.** Plans, does one move, checks which platform she landed on, and replans from there when it isn't the planned one. It hunts on the way: before each move, any monster a star reaches from where she stands, on any platform, is shot dead first. That's up to 5 per platform, so respawns can't hold her up, and it's turned off while she goes to rest. It also shoots any monster in the way of a move along a platform. A hit during a move, seen as lost HP, means she may have been knocked back: it replans from wherever she is, without counting a miss. It can stop early once a target is in reach. If her minimap dot disappears, for example under another program's tooltip, it moves the mouse inside the game to clear it. |
| `tour.py` | **Hunting tour.** Visits spawn platforms in random order. For each one, it picks where to shoot from: a safe platform with at least 40% of the hunting platform in reach, such as platform 14 above 17, or else the platform itself. A platform is safe when no monster spawns on it and none walks a platform less than 55 units underneath; platform 0, 42 units above platform 1's pigs, isn't safe. On the way, it starts shooting as soon as a target is in reach. At a safe spot it waits up to 10 s for monsters to respawn or walk into reach. If they stay out of reach, it hunts on the platform itself. After each round, it collects drops that fell out of reach. It only visits platforms where every spawn is a monster `hunt` can recognize. When HP is low, it rests on the chair at the nearest safe platform, and it ends standing by there. |

In game, on Thicket Around the Beach II:
- **Navigation:** 21 moves in a row across every move type without a miss, and a 17-move climb from the basin to the top. The fixes that made that work are in `route.py`: the run-up, simulated landings, and the measured walking model.
- **Tour:** visited the basin, platform 7 and platform 21 in random order, made 5 kills (EXP +64, matching the count), rested from 81% to 97% HP on a safe platform, and stood by.
- **Upper platforms:** with Orange Mushrooms recognized, two more tours covered 1, 2, 7, 18, 34 and 17. All 4 stops of the second were reached with no navigation misses, for 10 kills including 2 Orange Mushrooms (EXP 700 → 813).
- **One plan per move:** running jumps from wherever she stands and grabbing ropes on the run. A 15-move route and a 3-move route ran with no misses. Over the tours below, 3 moves landed somewhere else, and each time the navigator replanned from there.
- **Shooting from other platforms:** from platform 14 at monsters on 17 below, from 3 at monsters on 1 above, from 11 at 12 below and from 15 at 18 below. Most kills took 2 regular throws.
- **Tours with these changes:** 7 tours of 2-3 stops. The best made 8 kills from 17 throws in 2 minutes with 4 loot pickups and no potions. EXP went from 813 to over 1,200 over the session.
- **Not yet checked on its own:** placing her from the minimap dot and the camera when her name tag is hidden. It was used now and then during the tours, but never compared against the tag.

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
- **`hunt` needs a 1920×1080 game window,** the size its templates were made at.
- **Each new monster type needs a colour gate** (`palette` in `mobs.py`) before its game sprites are used. Its templates come from the game data automatically.
- **Her name tag can be hidden** behind screen widgets at the map's edge, like the party HP box. She's then placed on screen from the minimap dot and the camera, which is only good to about 11 map units.
- **Camera correction needs ground with a pattern** under her, like a girder or tiles. Plain ground can't be tracked.
- **Swing detection is tuned to the claw's white/lavender swing effect.** Other weapons or skills may need `SWING_MIN_PIXELS` in `track.py` adjusted.
- **Lucky Seven costs about 7 MP per cast.** A 32-shot test uses about 220 MP. Use `--regular-attack` for long runs. Hunting uses the regular attack unless `--skill` is given.
- **How far above or below a star reaches** beyond 60 units isn't measured. `hunt shots` sums up the shot log. A hit is read from the target's HP bar, which is sometimes missed, so kills are the surer sign.
- **Other players** hunt this map too. Platforms are often empty, and stops then end with no kills.
- **EXP from anything else** counts as a kill if it comes just as a target vanishes. In a party, that includes EXP shared from other members' kills.
- **Don't leave her standing on a spawn platform:** monsters walking into her pushed her off platform 12 while she waited there.
