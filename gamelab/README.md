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
- **Plans can be cut short:** when `lab/inbox/<plan id>.cancel` appears, it stops pressing and lets go of every key it holds. A plan it hasn't started yet can be withdrawn: it renames each plan as it takes it, so the client can tell. The navigator uses this when she's hit or off course mid-plan.
- **Plans can be queued:** a plan can say how long it lasts (`end_ms`), so the next one starts the moment it ends. Keys are still all released between plans.
- **Stops on the Lie Detector:** the player checks the game screen 4 times a second, idle or not. It recognizes the countdown window (yellow "LIE DETECTOR" title), the mouse mini-game ("The Lie Detector Mini-game: use the mouse to follow the fading figure") and the Truth Booth that a failed test sends her to. When one appears, it:
  1. stops all input within about 10 ms, mid-move included, and releases any held keys;
  2. writes `lab/ALARM.json` first, then saves a screenshot (once, exiting mid-save left neither);
  3. pages whoever is at the PC with alert sounds and a Windows notification that doesn't take focus from the game;
  4. shuts down, once the watchdog has finished recording and paging.

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
| `mobs.py` | Finds snails, slimes, pigs and Orange Mushrooms. It matches each monster's body-colour silhouette first, then checks the best spots with masked colour templates. About 100 ms for the fight's band around her, 300 ms for the whole screen. The first pass ignores windows with almost none of the monster's colour: their normalised score is noise that can reach a perfect 1.0. The frame is also cut to an even size so the half-size mask is an exact shrink. An odd size (the fight's band was 151 rows) smeared faint colour into blank windows. In the basin, those fake matches took all 12 checks per monster type, so the fight never saw the snails and pigs at her level while a whole-screen look did. That's likely also why one pass over the whole screen once missed three slimes in plain view; whole-screen looks still go band by band. |
| `sprites.py` | Templates for the monsters on the current map, made from the game's own sprites (every stand, move and hit frame) at their on-screen size. They're added to the screenshot cut-outs in `assets/mobs/`, which still catch monsters half hidden in plants. Built on first use and kept with the other extracted data (git-ignored). |
| `shots.py` | Where a star reaches, and where to stand to shoot a platform: the spots on other platforms that have the most of it in reach, safe ones first. Also reads a monster's HP bar. |
| `hunt.py` | `Eyes`: her name tag (full resolution), HP/MP from the bar colours, her minimap dot, and the monsters in her shooting band. `Hunter`: places her and the monsters on the map, and runs the objectives. |

Her position comes from the minimap dot, which is always visible. Her place on screen comes from her name tag. When the tag is hidden, for example behind the party HP box at the map's edge, it comes from the dot and the camera instead. The camera keeps her at a fixed spot on screen except near the map's edges, where it stops, and it learns that spot from the looks where the tag is visible.

What an objective does on its own:
- **Regular attack:** throws with the plain attack key. Lucky Seven costs MP and isn't needed on these monsters (`--skill` uses it anyway).
- **Range:** a regular star flies 400 map units (560 px) and hit a pig 388 units away. She throws as soon as the target is within 340 units, leaving room for it to walk during the half-second wind-up, and only walks closer when it's farther than that. Closer than 90 units the claw swipes instead, for 9-11 damage against 30-70 for a star. There she jumps and attacks in mid-air, where it's always a star (the macro's F1 combo: the attack 100 ms after the jump key).
- **Other platforms:** stars pass through platforms, so targets on a platform up to 60 units below or above her are shot too (both measured: a pig 60 below hit, a pig 60 above killed). They can't walk over to her.
- **Placing monsters:** a monster seen on screen goes on the platform its kind spawns on. Feet read from the screen can be 20 units off, which once put pigs "on" the safe platform above them, and she backed away from them and off it.
- **Spacing:** steps away from any monster on her own platform within 110 units, unless that walks into another one or off her platform. A monster between her and where she needs to stand is shot first. Cornered with one too close to throw at, she jump-throws at it from where she is. Jumping over it put the rest of the group on her other side.
- **Knockback:** a monster touching her pushes her away from it. The hit log (`lab/hits.jsonl`) records each push while no key was pressed: 20-21 units on a wide platform, and 82 and 109 units on a narrow one, which knocked her off. With monsters on her platform she stays 50 units from its ends, and a fight that ends on another platform stops the objective, so the tour sends her back.
- **Facing:** she only taps an arrow to turn when she isn't already facing the target. Every tap also walks her a few units, and nine taps in a row once walked her off the end of her platform.
- **Edges:** never walks off her platform, and skips targets she can't reach from it.
- **Fighting a group** (`clear`, what tours use): a star hits the nearest monster in reach on the side she faces. So she throws toward the side with the most monsters in reach, and keeps working the side she faces while it has any. Turning after every throw spread the damage over both sides, and once she killed one of two, then crossed over to the other side of the second. Throws come one per attack cooldown, with a look during each cooldown. She never walks past a monster.
- **Hidden monsters:** a monster she was throwing at is still thrown at for 1.8 s after it drops out of sight. Hit, it flinches under its damage numbers, and the detector loses it for 1-3 s. She used to stop throwing until it showed again, and once took 8 throws and 16 s over one pig.
- **Waiting:** with nothing in reach she walks just close enough to the nearest monster. With monsters seen but out of reach, she waits up to 2 s from a safe platform (1 s on the hunting platform) for them to walk in. With none seen at all, she moves on after 0.4 s.
- **Pick-up:** whenever she walks or runs, in fights and on routes, she presses pick-up every 0.15 s, so she loots whatever she passes over.
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
| `route.py` | **Moves and routes.** Builds every move the physics allows: rope climbs and climb-downs, and jumping up through a platform overhead. Running jumps (from `jumpSpeed` 555 and gravity 2000: 77 units high, 555 ms in the air, at her measured 149 units/s) are tried from every 20 units along each platform, not just its ends; a player jumps from wherever that reaches another platform, up, across or down. Up to 3 takeoffs are kept per landing platform. A jump is kept only if it still lands on the same platform when she leaves the ground 12 units early or late, or 10% slower: a jump meant for the basin once landed on a platform whose edge its arc passed close by. Walking off an end, and dropping through (↓ + jump), complete the set. Walking off wins over a running jump down to the platform it reaches anyway, which would be a needless jump. An end with a cliff dropping from it (platform 34 down to 36) can be walked off. The simulation used to start just inside the end and bump into the cliff's top, which left out 42 of the map's 63 walk-offs. Dijkstra then finds the quickest route. The cost of a move includes a small allowance for misses, measured on chained moves. It used to be 1.5 s per jump, from when she moved one move at a time, and that made drops win too often. |
| `flow.py` | **Moving like a player.** Turns several route moves into one continuous key timeline, predicted with the game's physics and her measured movement. She keeps running through landings, jumps on the way, and grabs ropes on the run. Each move has a few variants: keep running on landing, land and stop, jump straight up from anywhere under the platform above, or step back for a run-up. The chain picks the variant the next move can follow on from. A monster in the way of a run is hopped over. By the body sizes in the game data, her jump can't clear a pig or slime outright, only a snail. Touches were counted by the purple MISS or damage numbers above her (`hunt.touch_popup`); she's a thief with high evasion, so a touch often shows MISS and does no damage. Timed with the top of the jump over the monster, 6 of 8 hops touched it, 4 of them for damage. So a hop is timed so she can't touch it on the way up: she leaves the ground just outside its reach, plus the distance it takes her to climb to its height. Its position comes from its speed measured between two looks. Any contact then comes past its middle, where the knockback pushes her on rather than back. Timed this way, 7 hops over monsters standing or walking at her took no damage (4 touches were MISS). Monsters walking away from her aren't hopped: she closes on them too slowly, and 2 of 3 such hops took damage. The chain waits for those, or goes around. Holds of the same key merge, so a second rope's up isn't let go by the first climb's release. Climbing over a rope's top is an animation of about 0.35 s. The next move's arrow goes down just before the top so she walks off as soon as it ends. Jumping off the rope sideways below the top doesn't help: the jump has next to no lift. Measured at 60 fps with the camera fixed: a standing start runs 120 ms behind steady speed at 148.5 units/s, and a jump lifts off 55 ms after its key. A drop starts falling about 0.28 s after its key. After the arrow is let go she stops about 27 units on, 0.3 s later. A jump pressed in the air is ignored, and a held jump key doesn't jump again on landing. Pick-up is pressed every 0.15 s through every run, so she loots what she passes. |
| `moves.py` | **One move at a time**, closed loop, for what a chain can't predict, such as climbing down a rope, which needs her lined up on its top. Also gets her off a rope she's hanging on: up onto the platform near its top, else a jump off sideways. |
| `nav.py` | **Navigator.** One look, then plan, then chain as many moves as are safe and predictable into one key plan. Like a player, she doesn't go the same way every time. Each trip weighs every move by a random 1 to 1.35, so routes close to the quickest take turns, and moves start from varied spots: a running jump 33-45 units before the end, drops and straight-up jumps anywhere sensible over the overlap. From platform 14 to the basin, two routes take turns; from platform 1 down to 34, 14 different routes came up in 40 trips. All were within 1-2% of the quickest. Going up there's one corridor, the stairs on the left, and every other way is 20-50% slower, so that stays the same. While that plan runs it looks around once, about 0.9 s before the plan ends, and plans the next chain from where this one will end. That chain is queued to start the moment this one finishes, so there's no stop between chains. Chains are capped at about 3.5 s so these looks come often. Looking late matters: from platform 34, the basin's slimes were under the status bar. Meanwhile it glances at her minimap dot and HP about 15 times a second (35 ms each). A hit, or her standing on a platform the running leg doesn't go to, cuts both plans short. She only counts as standing on a platform when her dot has stopped at its height: mid-jump, the dot passes close under the platforms above, and those used to read as her being on them. That catches, for example, a drop that fell one platform too far before the next drop's keys fire. After a cut it does a quick local replan: it waits for the knockback to end (about 0.3 s), takes one fast look, and builds a new chain from there with the monsters it already saw, with no slow look. Without `--fight` (plain `nav`), she doesn't attack. The look before each chain covers the whole screen. A leg that a monster could walk into ends the chain: that move costs extra in the planner so a way around is taken, else she waits for the monster to move off, but only where none can reach her. Monsters are placed by their body width from the game data and their walking speed, and in the direction they were seen going. Jumping over pigs and Orange Mushrooms doesn't work: they're 67 and 64 units wide, while her jump stays above their height for only about 52 units. With `fight` (hunting tours), she fights instead. Before each move, she fights a short burst at whatever a star reaches from where she stands, on any platform: up to 3 kills, never walking after them, and done once nothing has been in reach for 0.3 s. That's up to 5 bursts per platform, so respawns can't hold her up, and it's turned off while she goes to rest. It also shoots any monster in the way of a move along a platform. Either way, a hit during a move, seen as lost HP, means she may have been knocked back: it replans from wherever she is, without counting a miss. It can stop early once a target is in reach. If her minimap dot disappears, for example under another program's tooltip, it moves the mouse inside the game to clear it. |
| `tour.py` | **Hunting tour.** Sweeps the spawn platforms: nearest next by route time, judged with a random ±25% so the sweep varies, then the order tidied to cut travel (2-opt). For 4 stops that's 23-24 s of travel, within a second of the best order of those stops and half the median of 4 picked at random (46-51 s). At a stop she fights, collects the drops, and moves on once a whole-screen look sees no monster left on the hunting platform. She used to go back to the safe spot after looting and wait for respawns, which looked idle and slowed the sweep. She goes back only for monsters still there, and not when the last fight threw nothing: in the basin, a whole-screen look saw pigs that the fight's own looks never did, and she went back 4 times without a throw. For each one, it picks where to shoot from: a safe platform with at least 40% of the hunting platform in reach, such as platform 14 above 17, or else the platform itself. A platform is safe when no monster spawns on it and none walks a platform less than 55 units underneath; platform 0, 42 units above platform 1's pigs, isn't safe. On the way, it starts shooting as soon as a target is in reach. At a safe spot it waits up to 10 s for monsters to respawn or walk into reach. If they stay out of reach, it hunts on the platform itself. After each round, it collects drops that fell out of reach. It only visits platforms where every spawn is a monster `hunt` can recognize. When HP is low, it rests on the chair at the nearest safe platform, and it ends standing by there. |

In game, on Thicket Around the Beach II:
- **Navigation:** 21 moves in a row across every move type without a miss, and a 17-move climb from the basin to the top. The fixes that made that work are in `route.py`: the run-up, simulated landings, and the measured walking model.
- **Tour:** visited the basin, platform 7 and platform 21 in random order, made 5 kills (EXP +64, matching the count), rested from 81% to 97% HP on a safe platform, and stood by.
- **Upper platforms:** with Orange Mushrooms recognized, two more tours covered 1, 2, 7, 18, 34 and 17. All 4 stops of the second were reached with no navigation misses, for 10 kills including 2 Orange Mushrooms (EXP 700 → 813).
- **Hops on routes:** across 4 routes (45 moves), she hopped 4 monsters in her way, with no hits and no misses.
- **Chains:** the 14-move climb from platform 32 to platform 3 took 33.4 s moving one move at a time, with about 4 looks per move. Chained, a 13-move route up took 19.0 s in 3 key plans with no misses. Planned on the move, 15-16-move routes take 20-24 s with no stops between chains. One of them spent only 0.9 s looking and 21 s moving. Most of the time is now moving: one route spent 1.8 s looking, 0.1 s thinking and 16.8 s moving. Over 10 test routes, a move landed elsewhere about once in 15, usually a drop or a straight-up jump timed off. Each time she replanned from where she was.
- **Shooting from other platforms:** from platform 14 at monsters on 17 below, from 3 at monsters on 1 above, from 11 at 12 below and from 15 at 18 below. Most kills took 2 regular throws.
- **Tours with these changes:** 7 tours of 2-3 stops. The best made 8 kills from 17 throws in 2 minutes with 4 loot pickups and no potions. EXP went from 813 to over 1,200 over the session.
- **Sweeps:** a 6-stop sweep before the fighting changes (hidden monsters, moving on after loot) made 4 kills (EXP +52) in 129 s, and stood still for 44 of the 108 s before her rest. With them, a 6-stop sweep made 7 kills (EXP +92: a slime, 5 pigs and an Orange Mushroom) in 114 s with no rest needed. She stood still for 33 s, half of that throwing. Four stops had no kills (other players hunt here): two were empty, and she left each about a second after the fight's last look.
- **Not yet checked on its own:** placing her from the minimap dot and the camera when her name tag is hidden. It was used now and then during the tours, but never compared against the tag.

```bash
python -m gamelab map
python -m gamelab nav 35            # add --fight to shoot monsters on the way
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
- **Don't leave her standing on a spawn platform:** monsters walking into her pushed her off platform 12 while she waited there. Standing in the basin, she was hit about every 10-20 s, pushed about 87 units each time.
- **Crowded destinations:** sending her without fighting to a platform full of pigs and Orange Mushrooms (17) took a minute of hits and replans.
