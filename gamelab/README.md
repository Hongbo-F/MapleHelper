# gamelab

gamelab drives the real game with exact key timelines, records it, and measures what happened. It started as the way MapleHelper's combo timing was measured: which way she jumped and threw, and whether the game ignored a key. A regular attack locks out jumps and attacks for about 730 ms. Quick taps right after takeoff were missed about 6% of the time.

On top of that it hunts on its own: it finds her and the monsters on screen, places them on the map, plans routes and fights, and sweeps a map platform by platform.

## Layers

The code is organized like a self-driving car's stack. Each layer is a package, and a layer only imports the layers below it (`tests/test_architecture.py` checks this).

```
  behavior      what to do next: the hunter (the agent), fights, looting, tours
     |
     +-- attack       which monster, from where, and how                    [pure decisions]
     +-- control      decisions turned into keys: actions, one move, a route; the key player
     +-- motion       motion planning, A to B: physics, routes, chained key timelines   [pure]
     +-- localization where she, the monsters and the drops are on the map (mined map data)
     +-- perception   pixels to what's on screen (no map knowledge)

  ui            the live mini-map window (reads telemetry)
  sim           an offline game: the stack runs against it with no game and no keys
  analysis      measuring combos in recordings
  shared        paths, screen geometry, left and right (directions), telemetry, the game window
```

| Package | Does | Main pieces | May import |
|---|---|---|---|
| `perception` | Pixels to what's on screen, in half-size screen px | `Eyes` (the only door to pixels): `look` gives an `Observation` (her name tag, HP/MP, the ground under her, her minimap dot, ropes, her EXP number); `monsters`, `items` and `bar` answer questions about a look's frame. `MobDetector`, `LootDetector`, `NameTag`, `LieDetectorSpotter` | shared |
| `localization` | Where things are on the map, in map units | `mapdata` (the map from the game's own data), `pose` (her position from the minimap dot, refined by ropes or her platform's end; the `Camera`), `Scene` (a look placed on the map), `Monster`/`MonsterTrack`/`Tracker` (monsters seen, told apart from look to look, and where they can be next), `Senses` (every look goes through it) | perception |
| `motion` | Getting from A to B | `physics` (jumps, falls and walking, measured), `route` (every move between platforms; Dijkstra), `flow` (several moves chained into one key timeline, hopping monsters) | localization |
| `attack` | Which monster, from where, how | `reach` (where a star hits), `stance` (throw, jump-throw, jump back, step away, close in), `spots` (where to stand to shoot a platform) | perception (monster names), localization, motion |
| `control` | Keys out | `Actions` (her key plans: throws, walking, pick-up, potions, the chair), `Mover` (one move, closed loop), `Navigator` (a route: chains, glances, cut-offs, replans), `client` (plans to the key player), `player` (the key player itself), `alarm` | perception, localization, motion |
| `behavior` | What to do next | `Hunter` (the agent: senses, actions, a looter, her vitals and health), `fight` (`clear`, `kill`, fighting on the way), `Looter` (sweeps, mop-ups, drops remembered), `Tour` | all of the above |
| `ui` | The mini-map window | `MapView` (draws a state; pure), `State` (the latest record per topic), sources (live, replay, a planned route), `MapWindow` | localization, motion, attack |
| `sim` | An offline game | `World` (her body, monsters, kills, drops), `FakeEyes`, `FakePlayer`, 51 scenarios, golden traces | everything but ui |
| `analysis` | Combo tests on recordings | `track`, `scroll`, `analyze`, `calibration`, `recordings` | perception, control |

How a look becomes keys:

```
Eyes.look -> Observation -> Senses.look -> Scene ─┬─> attack.stance ─────────────> Actions ─┐
   (perception)            (localization)         └─> motion.route / flow -> Navigator / Mover ─┴─> client -> player -> the game
                                         after each look: the hunter's vitals (EXP gains, HP) and telemetry
```

The loops live in `behavior/fight.py` (a fight), `control/navigator.py` (a route) and `behavior/tour.py` (a sweep).

## How the keys get to the game

MapleStory runs as administrator, and Windows blocks key presses from normal programs into admin windows. So one small process, the **player** (`control/player.py`), runs elevated, and everything else runs normally and talks to it through files in `gamelab/lab/`. It reads plans (JSON) from `lab/inbox`, presses keys only while the game is in front, records the game window if asked, and writes results to `lab/outbox`. `control/client.py` hands it plans and waits for the results; `control/timeline.py` runs MapleHelper's own combo code on a virtual clock, so tests play exactly what the helper plays.

What keeps the player safe:
- **No code from plans:** it runs no code it receives; plans are data.
- **Her keys only:** it only presses the keys `config.toml` `[keys]` binds to her actions, plus `\` (key bindings window) and Esc (close windows). Enter is never pressed, so nothing typed in chat can be sent.
- **No clicks:** it can move the mouse inside the game window to hover over things, but it cannot click.
- **Game in front only:** it stops pressing keys if the game leaves the foreground.
- **Fixed at start:** it loads its code and key list once, when it starts, and loads as little of gamelab as it can (a test checks).
- **Stops itself:** it exits when `lab/STOP` appears or after 3 idle hours.
- **Plans can be cut short:** when `lab/inbox/<plan id>.cancel` appears, it stops pressing and lets go of every key it holds. A plan it hasn't started yet can be withdrawn: it renames each plan as it takes it, so the client can tell. The navigator uses this when she's hit or off course mid-plan.
- **Plans can be queued:** a plan can say how long it lasts (`end_ms`), so the next one starts the moment it ends. Keys are still all released between plans.
- **Stops on the Lie Detector:** the player checks the game screen 4 times a second, idle or not. It recognizes the countdown window (yellow "LIE DETECTOR" title), the mouse mini-game ("The Lie Detector Mini-game: use the mouse to follow the fading figure") and the Truth Booth that a failed test sends her to. When one appears, it:
  1. stops all input within about 10 ms, mid-move included, and releases any held keys;
  2. writes `lab/ALARM.json` first, then saves a screenshot (once, exiting mid-save left neither);
  3. pages whoever is at the PC with alert sounds and a Windows notification that doesn't take focus from the game;
  4. shuts down, once the watchdog has finished recording and paging.

  It never tries to answer the test. Until you run `python -m gamelab clear-alarm`, every command that sends keys or brings the game to the front refuses to run (exit code 3), including starting the player again (`spot`, `minimap`, `sim`, `map --offline` and the recording tools still run). When Claude is driving gamelab, it also pushes the alert to your phone.

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

| Command | What it does |
|---|---|
| `hunt look` | What she sees now: her map position, HP/MP, room to walk on her platform, the monsters in her shooting band with their platform and whether a star reaches them, and the drops on every platform in view (annotated `lab/hunt_look.png`); sends no keys |
| `hunt clear --ground 17 --kills 6` | Shoot the monsters on platform 17 (and any on hers), looting kills that fall on her platform |
| `hunt kill --near -200` | Kill one monster (the one closest to that many map units away) and loot it |
| `hunt shots` | Hits and misses of every throw logged so far, by how far above or below her the target was |
| `map`, `map --offline` | The map's platforms and her position on them (`lab/map_<id>.png`); sends no keys |
| `nav 35`, `nav 35 --fight` | Go to platform 35; with `--fight`, shoot monsters on the way instead of avoiding them |
| `tour --stops 3 --kills 5` | A hunting sweep of the map, then rest and stand by |
| `minimap` | The live mini-map window, following the run in progress; see below |
| `minimap --replay FILE`, `minimap --plan 36 2` | Replay a recorded run, or show the route between two platforms; no game needed |
| `sim SCENARIO --speed 4` | Run the stack offline on the simulator, recorded for the mini-map; `sim list` names the scenarios |
| `do walk left 150`, `do climb_rope 300`, `do pick_up 2`, `do rest 40` | Run one move from `maplehelper/actions.py` in the game, recorded |
| `act right*800 alt 500 @960,540 --record` | Ad-hoc steps: `KEY` tap, `KEY*MS` hold, `KEY+`/`KEY-` press/release, `MS` wait, `@X,Y` move the mouse |
| `combo NAME --regular-attack` | A combo test, throwing with the plain attack key (no MP) |
| `track RECORDING`, `sheet RECORDING --around-her`, `check RECORDING`, `list` | Her path through a recording; a zoomed contact sheet; whether keys went out on time and the game kept a steady frame rate; saved recordings |
| `lock attack 700 740 780` | Whether a jump that many ms after an attack goes through |
| `snap` | Screenshot of the game in `lab/snap.png` |
| `spot`, `spot --image FILE` | Whether the Lie Detector is on screen now (or in a screenshot); sends no keys |
| `clear-alarm` | After you've answered a Lie Detector yourself: lets gamelab run again |

## The mini-map

`python -m gamelab minimap` opens a window that draws the map from the game data, and on it what the stack sees and plans, as it happens:

- **Localization:** her position (yellow; a magenta ring when placed from the minimap dot because her name tag is hidden) with the way she faces and her trail, HP and MP, and how her position was worked out (rope, edge or dot).
- **Perception:** the monsters in the last look, coloured by kind, with their HP bars once hit; the drops seen on the ground (yellow), and those remembered for later (hollow squares).
- **Prediction:** where each monster can be in the next second, as the move planner sees it (red bars).
- **Motion planning:** the planned route (cyan: arcs for jumps, arrows for climbs and drops) and the chain of moves in the running key plan (magenta: where she runs, and where each move lands).
- **Attack:** where a star reaches from her (the band either side, brighter on the side she faces), the monsters the last throw aimed at (red rings), and the fight's last decision.
- **The tour:** its stops in order, the current one, the spot she shoots it from, the safe platforms (teal) and the hunting grounds (bright green). Below the map, the latest log lines.

Hunting commands (`hunt clear`/`kill`, `nav`, `tour`) record what they see and plan to `lab/telemetry/<time>-<command>.jsonl` (`gamelab/telemetry.py`). The window follows the newest of those files, so it can be opened before or during a run. Live, the window never takes focus from the game: the player only presses keys while the game is in front, and screen captures see whatever is on top. So it goes beside the game window when the screen has room, and clicks on it don't activate it. A replay (`--replay FILE`, also last night's `lab/sweep_track.json`) is an ordinary window: space pauses, + and - change the speed, and the arrow keys jump 10 s.

## Offline: the simulator and golden traces

`gamelab/sim` is a small game that the whole stack runs against, with no game and no key player. `FakeEyes` answers perception's questions from a simulated world, and `FakePlayer` runs key plans on it on a fake clock. Her body follows the game's physics (the same jump arcs as the planner) and her measured walking and climbing; monsters walk their platforms, touch her (half of the touches are evaded, as measured), die from stars, give EXP and drop loot. 51 scenarios cover navigation (every kind of move, monsters in the way, fighting on the way, cut-off plans), single moves, fights, kills, mop-ups, a full inventory, tours and rests.

- `python -m gamelab sim tour_run --speed 4` runs a scenario and records it for the mini-map: open `python -m gamelab minimap` first to watch it as it happens.
- `python -m gamelab.sim.golden check` reruns every scenario and compares it with the golden trace recorded earlier: every key plan, log line and result. A refactor that leaves those byte-identical hasn't changed what she does. `record` rewrites the traces (after an intended change). The traces hold map data, so they stay in `lab/golden` (git-ignored).

The move into the layers above was checked this way: all 50 traces stayed identical, apart from two counters that were never read (`Report.out_of_reach`, `Hunter.skipped`), which were removed.

Tests can't press keys, page anyone or write under `gamelab/lab` (`tests/conftest.py` stops them), and the simulator refuses to run unless its fakes are in place.

## Perception

| Piece | Does |
|---|---|
| `mobs.py` | Finds snails, slimes, pigs and Orange Mushrooms. It matches each monster's body-colour silhouette first, then checks the best spots with masked colour templates. About 100 ms for the fight's band around her, 300 ms for the whole screen. The first pass ignores windows with almost none of the monster's colour: their normalised score is noise that can reach a perfect 1.0. The frame is also cut to an even size so the half-size mask is an exact shrink. An odd size (the fight's band was 151 rows) smeared faint colour into blank windows. In the basin, those fake matches took all 12 checks per monster type, so the fight never saw the snails and pigs at her level while a whole-screen look did. That's likely also why one pass over the whole screen once missed three slimes in plain view; whole-screen looks still go band by band. |
| `sprites.py` | Templates for the monsters on the current map, made from the game's own sprites (every stand, move and hit frame) at their on-screen size. They're added to the screenshot cut-outs in `assets/mobs/`, which still catch monsters half hidden in plants. Built on first use and kept with the other extracted data (git-ignored). |
| `loot.py` | Finds drops on the ground: what this map's monsters drop (Squishy Liquid, Slime Bubble, Leather, Pig's Head, Pig's Ribbon, Orange Mushroom Cap, Snail Shell), the potions monsters this level drop (Red, Orange, White and Blue Potion), and bronze and gold mesos (3 spinning frames each). Templates are the game's own item pictures at their on-screen size, scaled by their opaque pixels only. A spot counts when the masked template matches (0.6) and its colours differ from the template's by at most 24 on average, checked where the picture covers whole pixels. The sea and island backdrop matched the Blue Potion's shape at 0.60-0.63, and UI buttons the White Potion's; they differed by 27-57, real drops by 12-19. Only a strip just above each platform is searched (drops lie on the ground), on 4 threads: about 30 ms for her platform, 140 ms for every platform in view. On 110 saved frames it found all 18 drops and nothing else. Drops lay on the ground about 2 s after her EXP went up, and monsters are drawn in front of them. |
| `nametag.py` | Her name tag, matched at full resolution (at half size the score swings between ~0.6 and ~0.9). Of the good matches in the play area, the one nearest where she was wins: her name also shows in the minimap, the bottom bar and the party list. |
| `hud.py` | HP and MP from the bar colours; her EXP number, which only changes when she gains EXP; a monster's HP bar; the purple MISS or damage numbers when a monster touches her. |
| `minimap.py`, `terrain.py` | Her yellow minimap dot; the ground under her; ropes and ladders on screen. |
| `lie_detector.py` | The Lie Detector's three forms (see above). |

## Localization

Her position comes from the minimap dot, which is always visible. The minimap is the map at 1/16 scale shown 1.40625× larger, so one minimap pixel is 11.4 map units; the dot is snapped to the platform she's standing on. With her name tag on screen, her x is refined from a rope in view (ropes are drawn exactly at their map x) or the visible end of her platform. Her place on screen comes from her name tag. When the tag is hidden, for example behind the party HP box at the map's edge, it comes from the dot and the camera instead: the camera keeps her at a fixed spot on screen except near the map's edges, where it stops, and it learns that spot from the looks where the tag is visible.

The map model (`mapdata.py`) comes from the game's own data, read with `gamedata` and kept off git: platforms (footholds joined end to end), walls, ropes and ladders, portals, monster spawns, EXP per kill and the minimap's scale, in map units (the screen shows them 1.40625× larger at 1920×1080).

A monster seen on screen goes on the platform its kind spawns on: feet read from the screen can be 20 units off, which once put pigs "on" the safe platform above them, and she backed away from them and off it. From look to look, a monster is the same one when it's the same kind within 85 units; the move planner also tracks which way each walks and how fast (`Tracker`), to tell where it can be next.

## Motion planning

| Module | Does |
|---|---|
| `physics.py` | Her jump (`jumpSpeed` 555, gravity 2000: 77 units high, 555 ms in the air) and fall, and where a jump or fall lands, simulated against the map's platforms and walls. Her measured movement: walking at 149 units/s (`WALK_SPEED`; 148.5 at 60 fps), a standing start 120 ms behind, a jump lifting off 55 ms after its key, 27 units of slide after the arrow is let go (`RUN_LAG`, `JUMP_LAG`, `STOP_RUN`), and how long to hold an arrow to walk a given distance. |
| `route.py` | Every move the physics allows: rope climbs and climb-downs, jumping up through a platform overhead, running jumps (tried from every 10 units along each platform: every 20 missed the 30-unit window of takeoffs from platform 4 up onto 2, so 2 had no route), walking off an end and dropping through (↓ + jump). Up to 3 takeoffs are kept per landing platform. A jump is kept only if it still lands on the same platform when she leaves the ground 12 units early or late, or 10% slower: a jump meant for the basin once landed on a platform whose edge its arc passed close by. Walking off wins over a running jump down to the platform it reaches anyway. An end with a cliff dropping from it (platform 34 down to 36) can be walked off; the simulation used to start just inside the end and bump into the cliff's top, which left out 42 of the map's 63 walk-offs. Dijkstra finds the quickest route; a move's cost includes a small allowance for misses, measured on chained moves. |
| `flow.py` | Turns several route moves into one continuous key timeline, predicted with the game's physics and her measured movement. She keeps running through landings, jumps on the way, and grabs ropes on the run. Each move has a few variants (keep running on landing, land and stop, jump straight up from anywhere under the platform above, or step back for a run-up), and the chain picks the variant the next move can follow on from. A monster in the way of a run is hopped over: by the body sizes in the game data her jump can't clear a pig or slime outright, so the hop is timed so she can't touch it on the way up, and any contact comes past its middle, where the knockback pushes her on. Timed this way, 7 hops over monsters standing or walking at her took no damage. Monsters walking away from her aren't hopped: she closes on them too slowly. Climbing over a rope's top is an animation of about 0.35 s; the next move's arrow goes down just before the top. Pick-up is pressed every 0.15 s through every run. |

## Attack

- **Regular attack:** throws with the plain attack key. Lucky Seven costs MP and isn't needed on these monsters (`--skill` uses it anyway).
- **Range** (`reach.py`): with Keen Eyes maxed (+200 range by the skill data), stars on her level have hit slimes 408, 410 and 452 units off and missed at 560 and 634 (`lab/range_probe.jsonl`), so the farthest hit counted on is 460. She throws as soon as the target is within 410 units, leaving room for it to walk during the half-second wind-up. Closer than 90 units the claw swipes instead, for 9-11 damage against 30-70 for a star.
- **Other platforms:** stars pass through platforms, so targets on a platform up to 60 units below or above her are shot too (a pig 60 below hit, a pig 60 above killed). Off her level a star needs more room sideways: the shot log hit monsters 60 above her 7 of 15 times at 150-199 units and 85% beyond 250; 60 below, 56% at 100-149 and 79% at 150-199. So one above has to be 200 units off, one below 150 (`reach.min_dx`). With one right under or over her, she steps along her platform until it's in reach.
- **A group** (`stance.py`): a star hits the nearest monster in reach on the side she faces, so she throws toward the side with the most monsters in reach, and keeps working the side she faces while it has any. Turning after every throw spread the damage over both sides.
- **Too close:** she jumps away from it, turning toward it in mid-air to throw. A star leaves about 520 ms after the attack key, in the air or not (recorded at 60 fps), so the key goes in right after lift-off and the star leaves just as she lands, the whole jump away (about 85 units). It needs 110 units of room behind her and no monster within 160 there; otherwise (cornered) she throws from a jump straight up. Under a platform within her jump height she doesn't jump to attack (on platform 20, five jump-throws at a snail put her up on 19); she steps away instead, or (cornered) throws from the ground.
- **Spacing:** she steps away from any monster on her own platform within 110 units, unless that walks into another one or off her platform. With monsters on her platform she stays 50 units from its ends: a monster touching her pushes her 20-21 units on a wide platform, and 82 and 109 units on a narrow one, which knocked her off.
- **Facing:** she only taps an arrow to turn when she isn't already facing the target. Every tap also walks her a few units, and nine taps in a row once walked her off the end of her platform.
- **Where to shoot a platform from** (`spots.py`): a spot on another platform she can stand on, either one no monster can reach or one only monsters not worth hunting reach (snails: she shoots those when they come close), with at least 30% of the hunting platform in reach. Of the spots about as good as the best, she takes the nearest: coming up from the basin she shoots 21, 18 and 12 from the platforms below them, and from above she shoots down.

## Control

- **Actions** (`actions.py`): every key plan she sends, blocking until the player has run it. Throws come one per attack cooldown. Whenever she walks, she presses pick-up every 0.15 s; on a kill's drops she stops and presses it 4 times (a press takes one item, a pig drops several).
- **One move at a time** (`mover.py`), closed loop, for what a chain can't predict, such as climbing down a rope, which needs her lined up on its top. Also gets her off a rope she's hanging on.
- **A route** (`navigator.py`): one look, then plan, then chain as many moves as are safe and predictable into one key plan. Like a player, she doesn't go the same way every time: each trip weighs every move by a random 1 to 1.35, and moves start from varied spots. From platform 1 down to 34, 14 different routes came up in 40 trips, all within 1-2% of the quickest. While a plan runs she looks around about 0.9 s before it ends and plans the next chain from where this one will end, queued to start the moment this one finishes. Chains are capped at about 3.5 s. Meanwhile it glances at her minimap dot and HP about 15 times a second: a hit, or her standing on a platform the running leg doesn't go to, cuts both plans short, and it replans from wherever she is. Without an escort she avoids monsters: a leg a monster could walk into costs extra so a way around is taken, else she waits for it to move off. With an escort (hunting tours), she fights instead (see Behavior). If her minimap dot disappears, for example under another program's tooltip, it moves the mouse inside the game to clear it.

## Behavior

- **The hunter** (`hunter.py`) keeps count after every look, whoever took it: a kill counts only when her EXP number goes up (counting a vanished target as dead was wrong about half the time: hit monsters flinch and hide behind damage numbers, and other players kill monsters too). It drinks an HP potion below `--potion-hp` and hands control back below `--low-hp`.
- **A fight** (`fight.py`, `clear`): one throw per attack cooldown, with a look during each. She never walks past a monster. A monster of any kind that comes close on her own platform is in her way and fought, snails included. Reaching a stop's kill count doesn't end a fight while monsters are still in reach. On a tour she also shoots monsters worth hunting on any platform in reach. A star hits the first monster in its path, whatever its kind, so kills are matched against everything in the path, not just what she was aiming at.
- **Hidden monsters:** a monster she was throwing at is still thrown at for 1.8 s after it drops out of sight (hit, it flinches under its damage numbers, and the detector loses it for 1-3 s). A kill is still seen fading out for about a second; what's seen where a kill just died is left alone for 1.2 s.
- **Waiting:** with nothing in reach she walks just close enough to the nearest monster. With monsters seen but out of reach, she waits up to 2 s from a safe platform (1 s on the hunting platform). With none seen at all, she moves on after 0.4 s.
- **Fighting on the way** (`FightOnTheWay`): before each move, a short burst at whatever a star reaches from where she stands, on any platform: up to 3 kills, never walking after them, up to 5 bursts per platform. It also shoots any monster in the way of a move along a platform. It's turned off while she goes to rest.
- **Loot** (`looting.py`): each kill's drops are swept when the fight is over, if they fell on her platform (it looks again until she has landed: the fight often ended mid-jump). After the sweep she looks again and goes back for drops still on her platform, standing right on each one, for up to 3 looks. Drops elsewhere are remembered for the tour: her kills', and any seen on other platforms. A remembered spot is forgotten only when two looks 1.5 s apart, 3 s or more after the kill, both see it empty. A drop she has stood on twice that's still there is left alone for 90 s: for mesos that means someone else's drop; for an item it may mean her inventory is full, which the pickup log (OCR) says, or 3 such items in a row. With it full she leaves items and still picks up mesos.
- **A tour** (`tour.py`) sweeps every platform worth hunting. A platform's worth is the EXP a visit is expected to give: at first 30% of its spawns' EXP (game data), counted as 2 visits, then blended with what each visit really gave (`lab/grounds.json`). Monsters giving under a quarter of the map's best EXP aren't hunted, only shot when in her way (snails give 2, against 12-15). Platforms expected to give under 15% of the best one's are skipped: by the data, 34 and 35 spawn slimes, but 11 visits found only snails there. The order: nearest next by route time, judged with a random ±25%, then tidied to cut travel (2-opt); after each stop the rest of the sweep is planned again from wherever she ended up. At a stop she kills all she can reach from her spot; if some are left out of reach, she moves to another spot that reaches them rather than walking down into them, and only then collects the drops (nearest by route first; drops of monsters shot on the way are kept for the next collection, and none starts a trip after 180 s). She moves on once a whole-screen look sees no monster left. When HP is low, she rests on the chair at the nearest safe platform (on the way there she still drinks potions, but low HP doesn't stop the trip), and she ends standing by there.

## In game, on Thicket Around the Beach II

- **Navigation:** 21 moves in a row across every move type without a miss, and a 17-move climb from the basin to the top. Chained, a 13-move route up took 19.0 s in 3 key plans with no misses (33.4 s one move at a time). Over 10 test routes, a move landed elsewhere about once in 15, usually a drop or a straight-up jump timed off; each time she replanned from where she was.
- **Hops on routes:** across 4 routes (45 moves), she hopped 4 monsters in her way, with no hits and no misses.
- **Shooting from other platforms:** from platform 14 at monsters on 17 below, from 3 at monsters on 1 above, from 11 at 12 below and from 15 at 18 below. Most kills took 2 regular throws.
- **Sweeps:** a 6-stop sweep made 7 kills (EXP +92) in 114 s with no rest needed. With long-range shots, all 9 stops in 350 s with a rest: 28 kills (14 at the stops, 14 on the way) and EXP 6,459 → 6,856; far kills' drops were collected 5-50 s later. After the jump-back timing change, a full 9-stop sweep made 28 kills (6 on the way), EXP 6,856 → 7,247 in 309 s, lowest HP 92%. In crowds it took 7-9 throws a kill: each star hits whichever monster is nearest in its path, so the damage spreads.
- **Loot:** recorded at 2 frames a second, the sweep after a fight picked up a slime's Squishy Liquid and two coins within 8 s, and the tour's loot collection picked up drops seen from another platform. With her Etc tab full, a Snail Shell stayed after she stood on it twice; she read the game's message in the pickup log and switched to mesos only.
- **Not yet checked on its own:** placing her from the minimap dot and the camera when her name tag is hidden.

## Found offline, not fixed yet

The refactor kept these exactly as they were; each is pinned by a golden trace, so fixing one is a deliberate change:
- A trip to drops can spend its whole 60 s shooting monsters on the very platform it's going to (`tour_hunt_basin`).
- Monsters shot on the trip to rest or to stand by aren't counted in the tour's `on_the_way` (`Tour.rest` and `rest_and_stand_by` don't add the trip's kills; `tour_rest` shoots 7 on its way to 32, and reports none).

Fixed since, each a deliberate change to the scenarios that pinned it (re-recorded, and their `expects` changed to pin the fix):
- **Resting at very low HP now reaches a safe platform** (`tour_rest_hp_low`). After a stop ended on "HP low" (below `--low-hp`), `Navigator.go`'s first health check ended the trip to rest at once, so she sat down where she was, among the monsters, and in the simulator died there. The trip to rest now checks only for potions (`Hunter.drink_if_low`): she drinks on the way, shoots what's in her way, and rests on the safe platform.
- **Drops noted during a loot trip are kept** (new scenario `tour_collect_loot_on_the_way`; `tour_run`, `tour_hunt_from_safe_spot`). `Tour.collect_loot` set the remembered drops to the rest of its list after each trip, so the drops of monsters shot on the way were forgotten (only a later whole-screen look found them again). They're now remembered for the next collection (or a fight's sweep) rather than gone after in the same one: going after them at once, every kill on a trip made another trip, and in the simulator three tour scenarios ran out of time shooting respawns. Collections are also capped (`LOOT_SECONDS`, 180 s: no trip starts later than that into one), because where respawns guard the drops, as on 18 below platform 16, going back for them never ended. In `tour_run` one trip now goes to a kept drop first; in `tour_hunt_from_safe_spot` the last collection goes after the kept drops on 18 until the cap (115 s more, 6 more items picked up, 15 more kills).
- **Kills made shooting a monster in the way of a move count in the trip's kills** (`nav_fight_basin`: 16 -> 18, all of the world's 18; `tour_run`: 74 -> 76 on the way). `FightOnTheWay.clear_way` now returns its kills, as `shoot` does.
- **`tour --seed` seeds the navigator's route choices too**, not just the sweep order, as the simulator does (no scenario goes through the command; a test pins it).

Checked and kept: after a plan is cut short, the quick look has no monsters, so fighting on the way (and `Tour.hunt`'s stop when a target is in reach) is skipped for one chain. That lets her move on from wherever a hit knocked her first. With a look that sees the monsters there instead, she fought wherever she was knocked to: `nav_fight_across`, 12 s from 12 across to 17, ran out its 180 s (30 kills, knocked off 12 times, once all the way down to 32), and `tour_hunt_platform_12` took 79 s longer.

## Limits

- **Recalibrate** after moving the game window, changing maps or changing resolution. Her name-tag template is kept between calibrations; `calibrate --fresh` rebuilds it, and should be run where no rope crosses her name.
- **`hunt` needs a 1920×1080 game window,** the size its templates were made at.
- **Each new monster type needs a colour gate** (a body-colour test in `PALETTES` in `perception/mobs.py`) before its game sprites are used. Its templates come from the game data automatically.
- **Her name tag can be hidden** behind screen widgets at the map's edge, like the party HP box. She's then placed on screen from the minimap dot and the camera, which is only good to about 11 map units.
- **Camera correction for recordings needs ground with a pattern** under her, like a girder or tiles. Plain ground can't be tracked.
- **Swing detection is tuned to the claw's white/lavender swing effect.** Other weapons or skills may need `SWING_MIN_PIXELS` in `analysis/track.py` adjusted.
- **Lucky Seven costs about 7 MP per cast.** A 32-shot test uses about 220 MP. Use `--regular-attack` for long runs. Hunting uses the regular attack unless `--skill` is given.
- **How far above or below a star reaches** beyond 60 units isn't measured. `hunt shots` sums up the shot log.
- **Other players** hunt this map too. Platforms are often empty, and stops then end with no kills.
- **EXP from anything else** counts as a kill if it comes just as a target vanishes. In a party, that includes EXP shared from other members' kills.
- **Loot templates cover this map's drops only** (`ITEMS` in `perception/loot.py`); another map needs its drops' names added. Equipment, scrolls, and bills and sacks of mesos aren't looked for; she still picks them up when she walks over them.
- **The full-inventory message** is "You can't get anymore items." It's in the client, not the data files, so `loot.py` matches that and the usual other wordings. Without the OCR packages, only the 3-in-a-row rule tells.
- **The simulator is a model:** its timings and physics are fitted to gamelab's own measurements, its monsters don't flinch, and its perception is exact unless a scenario adds noise. Use it to check that a change does what was meant, then confirm in game.
