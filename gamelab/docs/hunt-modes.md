# Design: hunt and transit modes

Status: steps 0-5 and 7 built in the simulator on 2026-10-19 (see "As built" below); step 6, the game trials, is still to do. It starts from commit 1e6ae23.

## In short

The hunter gets two explicit modes instead of six kinds of trip:

| Your words | In this design |
|---|---|
| attack mode: follow the motion plan and look for monsters to kill along the way; the plan changes as monsters respawn | **hunt**: one loop that decides what to do after every look (rules below), so a respawn changes the next step within a second or two |
| non-attack mode: follow the motion plan and avoid getting hit | **transit**: going to rest or to stand by. She doesn't stop to hunt, and only shoots what blocks her path or comes too close |
| loot picked up on the way | pick-up pressed on every walk and every chain of moves, and a stop on piles she passes on her platform (rule 5) |
| loot gone after once all monsters are killed | the lull rule (rule 8): in the gap before respawns, she collects piles she can reach in time |
| loot gone after when it's about to expire | the urgent rule (rule 6), placed above moving to a better spot and above waiting |

A safety layer sits above both modes and can switch the mode at any look.

There is no separate "hunting trip". She stays at a hunting ground while it pays (rules 1-9) and leaves by the rate rule (rules 10-11) ("Staying and leaving"). That replaces the stops, loot trips and fights on the way of today's tour. The tour's sweep stays as the order in which grounds are tried, not a list of stops she must finish.

## As built (2026-10-19)

Steps 0-5 and 7 are in place, offline: `tour --loop --minutes N` (`behavior/hunt.py`), the ledger and drop store (`behavior/ledger.py`), the explicit trip modes and the stepped navigator, the results harness (`sim/harness.py`), 10 new scenarios (`loop_*`, `tour_rest_avoid`, `tour_rest_no_escort`), and `tour --rest-avoid`. Step 6 needs the game.

Where the loop differs from the design below, each change was made because the simulator showed the design's version failing:
- **Urgent piles (rule 6) come before shooting (rule 3)** at a ground, with a 30 s margin (not 10 s), and only for piles within 12 s of travel. Below shooting, where respawns never stop, a quarter of the drops lay until they vanished. Farther piles wait until she leaves. One 84 s round trip for a pile took a quarter of a run.
- **Back to the spot from any platform** (rule 2, widened): after a pile on another platform, she goes back to her spot unless the ground is in reach from where she is. Without this, she hunted the pile's platform under the old visit and never left.
- **A ground hunted on itself (the basin) counts as reached anywhere on it.** Waiting to reach the spot's x, close monsters interrupted her for good, and the visit never began.
- **Lull loot (rule 8) only while a kill of hers on the ground is due back.** With nothing due back, the ground is empty and she leaves (the fast exit); going after piles from an empty ground, she never left it.
- **Leaving (rules 10-11):** the piles left when she decides are the only ones gone for, and no new urgent piles are taken while leaving. The decision itself is re-checked at every pass: kept once made, a weak reason ("nothing came into reach") lost the respawns that came after it.
- **Rule 5 also while travelling**, as the In short table says: a pile she passes on her platform.
- **The objective's clock is wall time** (60 s, suspensions and rules 1-4 included), as designed. Counting only an objective's own trips, one run tried for 270 s to reach a spot and never got there.
- **The stuck guard** counts a pile she went for that a look then sees gone as a pick-up, not the attempt.
- **From code reviews** (two reviews, then two rounds of checked fixes):
  - a visit's length is blended with a prior like its EXP (one empty ground left in 0.6 s looked like the best ground);
  - kills count for a visit only after she got there, and on its ground or from it and its spot (unplaced ones by where she was);
  - a rest closes the visit and starts a new, undecided one;
  - she shoots (rule 3) only where she can stand, or on the ground and its spot;
  - while leaving: no high ground, back to spot, better spot or new urgent piles (at the ground, on the way, or as a reason to stop a trip); away from the ground, the decision stands;
  - "nothing left here" (the fast exit and its 2-minute skip) only from looks that cover three quarters of the ground, still empty 3 s later: hit monsters hide for 1-3 s, and single looks skipped busy grounds; "nothing came into reach" only with nothing due back, and from where the ground is seen;
  - a whole-screen look that didn't place her isn't used, and with no platform the next ground is chosen by value alone;
  - potions are out after two misses in a row, and back if one works; she still presses the key (a false alarm costs nothing that way);
  - a pile trip ends when the pile is seen gone or can't be picked up (not at its guessed age: that lost 6% of the drops in the simulator, where drops last twice as long);
  - each rule is counted once per pass.
- **Built since (2026-10-20):** chains follow on in every trip mode (README, Control: "A route"). The navigator's look about 0.9 s before a chain lands asks the loop's `_interrupt` (rules 1-4, an urgent pile, low HP) on where the chain ends; if nothing fires, the next chain is queued from a look placing her once she has landed, else she stops there as before. A trip starts from the loop's last look when it's fresh.
- **The grounds worth a visit by their EXP a second** (since the review of the three tracks' merge): the loop kept the ones the tour's sweep keeps, worth 15% of the best one's EXP a visit, but the loop's visits last as long as a ground gives (15 s to 600 s), and a ground's EXP a visit grows with them. One visit of 351 s to the basin made it worth 136 a visit and put every other ground under the cut (the best, 17, at 19.5), and the loop, leaving the basin empty, went straight back to it, twice, for the rest of the run (harness seed 7: 1,150 of its 1,200 s there, 57.6 EXP a minute; master's code does it on seeds 11 and 19). It now keeps those worth 15% of the best one's EXP a second (`Tour.hunting_grounds(visit_s)`, with `Hunt._visit_s`). Over 30 harness seeds three runs change: seed 7 57.6 -> 62.8, seed 5 114.5 -> 108.6, seed 16 131.3 -> 130.7 EXP a minute; the mean 123.1 -> 123.0.
- **Not built:** the whole-screen loot look 3 s after each kill, and "the same objective dropped 3 times" in the stuck guard.

Tried and dropped, because they were worse in the A/B runs: no shooting at all once she has decided to leave (one seed halved), skipping piles among monsters on platforms she can't stand on (the `plain` world's drops share fell from 91% to 77%), and an objective clock without suspensions.

**Results** (preset `game`, 20 simulated minutes a seed; the README has the table): the loop makes 114.2 EXP a minute against the fixed tour's 84.8 on 10 seeds (117.7 against 85.7 on 20), better on 19 of 20 seeds, with the same share of its drops picked up (95.8% against 95.6% on 20 seeds; 250 drops a run against 185), less HP lost and no deaths. The one point of the bar it misses is seed 7, 35% worse: knocked off the spot on 16 again and again, it spends its time climbing back. With `--think 1` it passes the whole bar (116.8 against 80.8). In the crowded `plain` world it makes 208 EXP a minute against 133, but picks up only 69% of its drops. Seed 7 and the drops share move a lot between near-identical variants, so differences of a few points are noise.

**Step 7:** with `--rest-avoid` (navigator mode `transit-avoid`), 40 trips to rest at 28% HP lost 1.6% HP a trip instead of 6.2%, in 6.1 s instead of 28.5 s, and nobody died (`python -m gamelab.sim.harness rests`). It passes its bar and stays an option until checked in game.

## Terms

- **Ground**: a platform worth hunting (today's hunting grounds: monsters giving at least a quarter of the map's best EXP spawn there).
- **Wanted**: monsters of the kinds worth hunting (pigs, slimes, Orange Mushrooms here; not snails).
- **Standable**: a platform she can shoot from safely: no monster can reach her there, or only unwanted ones can (`spots.spot_ok`).
- **Spot**: where she stands to shoot a ground, on a standable platform (`spots.spot`, `spot_for`), or the ground itself when no standable spot reaches it, like the basin (36).
- **Objective**: what she's heading for right now: a spot, a pile, or the next ground.
- **Pile**: drops lying together (within 15 units).
- **Value map**: the per-ground stats in `lab/grounds.json`: visits and the EXP they gave.
- **Ledger**: the list of her kills (kind, platform, time, ground), one for the whole hunt.
- **Backstop**: a generous cap that should never decide anything. It logs when it fires.

## Why

Today a tour is a sequence of trips, and each kind has its own rules:

| Trip | Fights | Loots | Ends on | Time |
|---|---|---|---|---|
| to a stop's spot | bursts on the way (3 kills or 15 s, 5 per platform), monsters in the way | in fights | arrival, or a target in reach where she can stand | the stop's time left, at least 20 s |
| the stop's fight | `clear` with finish and far | sweeps in the fight | kill count, patience, low HP, knocked off | the stop's time left, at least 5 s |
| loot trip | bursts and monsters in the way | at the drop | arrival | 60 s; no trip 180 s into a collection |
| guard fight at a drop | `clear`, 2 kills | after it | kill count, patience, low HP | 20 s |
| to rest | monsters in the way only | in those fights | arrival (low HP doesn't end it) | 180 s |
| to stand by | bursts and monsters in the way | in fights | arrival | 180 s |

The bugs fixed this week lived in the gaps between these rows:
- the trip to rest ended on low HP;
- loot trips forgot what their own fights noted;
- going after those drops at once never ended;
- kills in the way weren't counted.

Two are still open:
- a loot trip to a platform full of monsters spends its whole 60 s shooting there, because the burst comes before the arrival check;
- kills on the trips to rest and to stand by aren't counted.

The simulator's `tour_run` (3 stops, 594 s) shows how the trips trap her. Every one of its 114 kills was made in the basin (36):
- 29 at the first stop;
- 41 on three loot trips to drops on the basin itself (the open bug: all three timed out);
- 35 while passing through the basin on the way to the other two stops (33 in bursts);
- 9 on the way to rest, which aren't counted (the second open bug, and the whole gap between the reported 105 and the world's 114).

The second stop reached its spot on 31, was knocked off into the basin by a snail 2 s later, and the trip back timed out fighting there. The third never reached its spot.

Elsewhere:
- **Stops overrun their time:** stops given 60 s ran 278 s (`tour_hunt_basin`) and 381 s (`tour_hunt_from_safe_spot`).
- **The value map undercounts:** it credits `tour_hunt_basin` with 19 of its 58 kills, because kills on the way and on loot trips belong to no ground.

So the problem isn't that she hunts on the way. It's that each trip has its own budget and its own idea of when to fight, and none of them can say "this has stopped paying, leave".

## Layers

```
  behavior   mission: grounds and their order (Tour: value map, sweep)
             safety: HP, rest, potions, stuck guards; may switch the mode at any look
             the hunt loop and transit (new: behavior/hunt.py)
  attack     pure decisions: stance, spots, reach            control   one step at a time: Navigator.step,
                                                                       Mover, Actions, the key player
                                motion     pure planning: physics, route, flow (chains)
                                localization, perception (unchanged)
```

Control may not import attack (`tests/test_architecture.py`). So the navigator asks behavior through a callback, as it asks today's `Escort`, at chain boundaries and on the look it takes during a chain.

## Safety, above both modes

Checked after every look, before the mode's own decision.

- **Lie Detector:** unchanged. The key player stops everything by itself.
- **HP:**
  - Below `--potion-hp` (0.45) she drinks, at most every 3 s.
  - Below `--rest-below` (0.5) she goes to rest at the next calm moment: nothing wanted in reach and no pile being picked up.
  - Below `--low-hp` (0.3) she switches to transit at once.
  - The goal is the nearest safe platform at least 120 units wide. Low HP is never a reason to stop that trip (the bug fixed in 1e6ae23).
  - She sits until 90% or 120 s, then goes back to hunting.
- **Potions running out:** a drink that brings no HP back within 2 s means none are left. She stops drinking, logs it, and from then on rests as soon as HP is below `--rest-below`, without waiting for a calm moment. Today this isn't noticed, because `Actions.drink` reports success whenever the key is bound.
- **MP:** none is used with the regular attack, which tours use. With `--skill` (about 7 MP a throw) an MP check would be needed; tours don't offer it.
- **Knocked off, falling, on a rope, dot hidden:** the navigator and mover deal with it as today. In hunt mode being knocked onto another platform ends nothing: the next decision starts from where she is, and rule 2 takes her back up if she's below her spot.
- **After a hit cuts a plan short:**
  - Until one chain has run, she may only answer a monster on her platform the claw would swipe (`stance.close_to`, by its size: a pig within 111, a snail within 96; 136 and 121 with the claw's old reach, 100 for every kind before) by stepping away from it, at most once, and doesn't shoot (a jump back throws a star).
  - Shooting from wherever a hit knocked her made a 12 s trip (`nav_fight_across`) run out its 180 s, and `tour_hunt_platform_12` take 79 s longer. Both stay pinned.
- **Stuck guard:** no kill, pick-up or new ground for 120 s, or the same objective dropped 3 times. She stands by on a safe platform and logs why.

## Transit

The goal is a platform and an x: a safe platform to rest on, or to stand by on.

Through step 6 of the plan, transit is exactly today's trip to rest:
- the quickest route, by the trip's random weights;
- no bursts at monsters in reach;
- a monster on her platform in the way of the next move starts a short fight (2 kills or 15 s) at anything in reach;
- close-quarters answers;
- potions but no low-HP stop.

The navigator's avoiding behavior also exists today, without the fighting escort, as used by `nav`. It sees the whole screen, routes around moves a monster could walk into, waits a little, hops over monsters and plans chains on the move.

Whether that is better when she's hurt is a separate question:
- **Against:** detours of up to 6 s a move, waits that happen near monsters, hops that can take contact.
- **For:** she isn't standing still to shoot.

Step 7 tests it with its own bar: HP lost on rest trips, and no deaths.

## Hunt

### A pass of the loop

1. **Look.** A look always covers the band around her (140 above to 75 below her feet; about 100 ms of detection in game) and her platform's loot strip (about 30 ms).
   - A whole-screen look (about 300 ms) comes when rules 1-5 don't apply, at most every 2 s, and before rules 10-11 are decided.
   - A whole-screen loot look (about 140 ms) comes before choosing among piles, and about 3 s after each kill (its drops land about 2 s after the EXP goes up).
2. **Update the world model:** tracked monsters (with their walking speed), the ledger, the drop store and the ground's rate.
3. **Rules:** the first rule that applies decides.
4. **One step:**
   - one throw, jump-throw or jump back;
   - one walk of at most 300 units, with pick-up pressed every 0.15 s;
   - one chain of moves (at most 5 moves, about 3.5 s, watched by glances, cut on a hit or off the path, as today);
   - standing on one pile for 4 presses.

**How fast a respawn changes the plan:**
- **Inside the band:** at the next pass, at most a throw's cooldown (about 0.8 s) or a chain later.
- **Elsewhere on screen:** within about 2 s.

**Chains keep moving.** In hunt mode the next chain is planned on the move, as without an escort today. Rules 1-4 are checked on the look the navigator already takes 0.9 s before a chain ends, and the queued chain is cancelled only if one of them fires.
- Today, with the fighting escort, every chain ends in a full stop: a 27-unit slide, two looks and a standing start. That costs a quarter to a half of a 3.5 s chain.

**The attack lock:** every plan that can press jump or attack waits out the 730 ms lock in one place, which both the navigator's chains and the mover's single moves go through. The lock's remaining time goes into the chain's start, so only its first jump waits.

### The rules

Her **targets**:
- wanted monsters in reach;
- any monster, of any kind, on her platform within 150 units;
- any monster, of any kind, between her and the start of her next move (±110).

These are today's rules (`_wanted`, `IN_HER_WAY`, `clear_way`).

| # | Rule | When | What she does | Bounds |
|---|---|---|---|---|
| 1 | Close quarters | a target on her platform the claw would swipe (`stance.close_to`, by its size: a pig within 111, a snail within 96; 136 and 121 with the claw's old reach, 100 for every kind before) | `stance`'s answer: jump back if there's 110 units of room and open sky; else jump and throw; else step back; else throw from the ground | always on, also during rules 5, 6 and 8; right after a cut, only stepping away |
| 2 | Back to the high ground | she's on a platform that isn't standable, and a standable spot reaches her current ground | one chain toward the spot, shooting only rule 1 and rule 4 targets | applies after a knock-off and after looting below a spot |
| 3 | Shoot | a target in reach, and she's on her spot or standable | `stance`: throw (the side she faces while it has targets), step out from under a monster right below or above her, or close in along her platform | walks of at most 300 (stance's own walks aren't capped today) |
| 4 | In passing | a target in reach while she's on another platform that isn't standable, or a monster in the way of her next move | throw from where she is; no walking after targets | 15 s of shooting per platform per objective, counted on the objective's clock |
| 5 | Pile here | a pile on her platform within 150, no monster predicted within 110 of it while she gets there, and no monster between her and it | walk to it, 4 presses | inside the end margins (50 units while monsters share the platform); on a platform that isn't standable, only with no monster on it |
| 6 | Urgent pile | a pile's remaining life is under the time to get there, plus 4 presses, plus 10 s | go for it | rule 1 stays on; she shoots only rule 4 targets on the way; at most 2 tries or 30 s per pile |
| 7 | Better spot | wanted monsters seen out of reach, and a standable spot reaches at least 2 more of them (or 1.5 times as many) at their predicted positions | one chain toward it | within 8 s of travel; not again for 10 s after arriving; at most 4 changes per ground |
| 8 | Lull loot ("all killed") | nothing wanted in reach or seen on this ground | piles she can reach and pick up before the next respawn is due | then rule 2 brings her back |
| 9 | Wait | monsters seen nearby, or one of her kills here due back within the respawn time + 2 s | stand still | idle time per position, reset by a throw or a pick-up; she stops waiting after 1 s on the ground, 2 s from a spot off it, 0.4 s with nothing seen |
| 10 | Loot elsewhere | the rate rule says leave, and piles are left on other platforms | go to them, most piles per second of travel first | each pile at most 2 tries |
| 11 | Advance | the rate rule says leave | one chain toward the next ground's spot; rules 1-4 checked on every chain's look | the objective's clock |

**Why rules, not a score:**
- The fight rules hold measured lessons: high ground, facing, room for a jump back, end margins.
- Weights for one big score would be guesses, and the simulator is too easy to tune them on: about twice the game's EXP a minute, with stars that never miss.
- Scores are used only where places are compared by time: spots, piles and grounds.

**Where the rules come from:**
- Rules 1, 3 and 9 are `clear`'s per-look steps (close quarters, throw, step from under, close in, patience).
- Rule 4 is today's fighting on the way (bursts and `clear_way`), with a budget instead of 5 bursts of 15 s.
- Rules 5, 6 and 8 replace the sweeps inside fights and `collect_loot`.
- Rule 7 is `spot_for` with a margin.
- `stance`, `spots` and `reach` are reused as they are.

### Objectives are suspended, not ended

When a higher rule fires, the current objective is paused, not dropped. It keeps:
- its route's random weights (drawn once per objective, as today once per trip);
- the pick of the next ground;
- its clock.

Time spent in rules 1-4 counts toward the objective's 60 s backstop and the ground's maximum. That keeps a shoot-respawns-on-the-way loop bounded, which today takes five caps.

A target that has just gone out of sight stays a target for 1.8 s. Hit monsters flinch and hide under damage numbers for 1-3 s in game. The simulator's monsters never hide, so a decider that drops targets at once would look fine offline and thrash in game.

### Staying and leaving a ground

Today a stop ends on its kill count, its seconds, a whole-screen look with nothing left, or a fight with no throws. The caps (`ON_WAY_TRIES`, `MAX_FIGHTS`, `GOES_PER_STOP`, the 60 s loot trip, `LOOT_SECONDS`) are backstops for trips. All of these are replaced by one rule:

Leave when `rate_here < best_next_value / (travel + visit_time)`, with:
- **`rate_here`:** the EXP she has gained since she took this position (far kills included) divided by the time there, plus a prior worth 20 s at the ground's expected rate.
- **`best_next_value`:** a ground's expected EXP per visit (the value map), so the right side is the best other ground's EXP per second including the trip there.
- **`visit_time`:** the average length of a visit. It isn't recorded today, so it's 60 s until measured visits replace it.
- **Minimum stay:** 15 s, or until 6 throws.
- **Margin:** she leaves only when `rate_here` is 20% below the right side, so two grounds can't swap back and forth.
- **Fast exit:** a whole-screen look sees nothing wanted that any spot of this ground reaches, and none of her kills here is due back within the respawn time + 2 s. Then she leaves at once. Empty grounds are common in game ("Platforms are often empty").

The value map keeps EXP per visit, and starts recording how long each visit lasted. Every gain is credited to the ground she's hunting. A ground she leaves by the fast exit is skipped for 2 minutes.

All numbers in this section and in the rules table are starting values, to be tuned in the simulator's A/B runs and checked in game.

### One minute on 16, shooting 17 (illustrative)

- **0 s:** She's on her spot on 16, a standable platform at the same height as 17. The band look sees two pigs on 17 in reach. Rule 3: she throws, one throw every 0.8 s.
- **6 s:** A pig goes down at x 950 on 17. The ledger places the kill, and the store expects its pile there about 2 s later.
- **8 s:** The second pig is killed. The whole-screen look sees nothing wanted left on 17.
- **8.5 s:** Rule 8, the lull. Both piles are about 3 s away and the first respawn is due at 14 s. She chains onto 17, walks over the piles with pick-up pressed, and stands on each for 4 presses.
- **13 s:** Rule 2: 17 isn't standable, so she goes one chain back to 16.
- **16 s:** A respawned pig is in reach: rule 3.
- **40 s:** The ground's rate is 0.6 EXP/s. The best other ground offers 52 EXP per visit over 15 s of travel plus 60 s there: about 0.7 EXP/s. 0.6 isn't 20% below that (0.56), so she stays.
- **55 s:** No kill since 40 s: the respawned pigs walk the far end of 17, out of reach of every spot. The rate is down to about 0.48, under 0.56, and no piles are left, so she advances (rule 11).

## Loot

**One drop store** replaces every place that writes the remembered drops today:
- `Looter.remember` and `note`;
- `collect`;
- `loot_pending`;
- `sweep_after`;
- the per-kill append in `clear`;
- `collect_loot`'s swaps.

Each pile keeps:
- when the kill happened (today a pile noted from a look gets the time it was first seen, so ages are underestimated);
- when it was last seen;
- whether it's hers (a kill in the ledger explains it);
- its status: expected (landing), seen, unpickable or taken.

The rules for forgetting stay: never before 3 s after the kill, and only after two empty looks 1.5 s apart, because monsters are drawn over drops. Any look that searches for loot updates the store, whoever takes it, through `Senses.listeners`.

- **Picked up on the way:** pick-up is pressed every 0.15 s on every walk and every chain's ground stretches. The mover's own walks (after a trip, lining up under a rope, single moves) press nothing today; step 1 adds it. Walking past a pile at full speed picks up only 1-2 of a pig's drops, so a pile on her platform gets a stop (rule 5).
- **All monsters killed:** rule 8, in the gap before the next respawn. With respawns about every 8 s per monster in the simulator (unmeasured in game), "all killed" is a short window, so the rule asks whether the pile can be reached and picked up before the next one is due.
- **About to expire:** rule 6.
  - The real lifetime isn't measured: the code assumes 90 s is still worth a trip, and the simulator keeps drops for 180 s.
  - Until it's measured, there is one constant in one place.
- **Guards:** a pile with any monster within 110 isn't a candidate, and that monster, of any kind, is a rule 3 target. No loot trip can turn into a 60 s fight: the time goes on the objective's clock.
- **Other players' drops:** piles no kill of hers explains are probably someone else's, and can't be picked up until their ownership runs out. They're skipped until then, and that time is unmeasured too. Today she travels to them and finds out after standing on each one twice.
- **Value:** 1 per pile, mesos and items alike, until prices are known.
- **Full inventory:** mesos only, as today.
- **Drops from kills on the way:** drops noted while going after other drops wait for the next pass of the rules. Going after them at once is what made every kill on a trip make another trip.

## Kills and reports

The ledger is separate from the targeting state.

- **The ledger:** one list of kills (kind, platform, time, ground) and the count of EXP gains already accounted for. It lasts the whole hunt.
- **The targeting state:** what she's thrown at, hidden targets, fading kills, the idle clock. It's per position, and resets on a new objective, a platform change, or after a hit.

Every EXP gain the hunter sees (one a look at most) goes into the ledger and is credited to the ground she's hunting:
- **A pile only when the gain matches a star:** the gain is matched against the paths of her throws in the last 1.2 s (520 ms to release plus the star's flight), with where she stood at each throw. Then its drops go to the store.
- **Otherwise no pile:** gains that match nothing (party EXP, a kill seen only from a look without monsters) count, but create no pile. A made-up pile out of view would never be forgotten, and the urgent rule would send her to it.

**One report for the whole hunt.** Today `Tour._fight` replaces the report at every fight, after its counts were copied. So kills seen in later looks land in a report nobody sums: that's how the guard fights' kills are lost. The trips to rest and to stand by lose theirs another way: their `trip.kills` is never added. HP lost (`damage_taken`) isn't added up at all today; the single report sums it.

## Rules that must survive

Each one was learned the hard way, and each gets a scenario in the loop's own set:

- Stand only where she's safe, or where only unwanted monsters can reach her (rule 2's "standable"). Stopping on the crowded platform 4 on the way to 2 cost 40 throws for 4 kills. And in `tour_run` a snail on a standable spot (31) knocked her into the basin: knocked off a spot, she goes back to it.
- Finish what's in reach before leaving a good spot; don't walk down into a group to loot.
- Throw the way she faces while that side has targets. Tap an arrow only to turn: nine taps walked her off a platform.
- No jumping to attack under a platform within 82 units overhead. Jump back only with 110 units of room inside the end margin, and no monster within 160 behind her.
- Keep 50 units from her platform's ends while monsters share it: knockback is 82-109 units on a narrow platform.
- Monsters in her way, or close on her platform, are shot whatever their kind.
- Monsters are placed only on platforms their kind spawns on: screen feet can be 20 units off.
- Off her level, keep the room sideways the star's search needs for the monster's kind (`reach.shot_dx`: 60 below, an Orange Mushroom 113, a pig 151, a slime 177; 60 above, 111-126 by the client's window, 191-207 by the logs' top it had before).
- She's on a platform only when her minimap dot has stopped.
- After a cut, move on before shooting (`nav_fight_across` arrives in about 12 s).
- Low HP never stops the trip to rest (`tour_rest_hp_low`).
- Every key plan stays cancellable, and any error cancels every plan sent.
- No plan that presses jump or attack goes out inside the attack lock.
- Every telemetry topic the mini-map reads keeps its meaning (see `ui/state.py`): session, log, event, scene, glance, drops, monsters, route, chain, stance, tour, trip. The navigator sends `monsters` today only without the escort; hunt mode must send it. `trip` is what clears the drawn route, so the loop sends it when an objective ends. `tour` gets `order` = the grounds visited and ahead, `stop` = the current ground, `spot` = the current spot, plus `safe` and `grounds` as today.
- A new topic, `hunt` {mode, rule, objective, rate here, rate to beat, piles with status and urgency}, shows on the mini-map why she does what she does.

## Ending, stopping, switching back

- **`tour --loop --minutes N`** hunts for N minutes, then goes to a safe platform and stands by, as `tour` does after its sweep. `--stops`, `--kills` and `--seconds` don't apply to it. The ground maximum and the rate rule do their job.
- **Ctrl+C, an error, or `python -m gamelab stop`** works as today: the navigator cancels every plan it sent, and the key player releases every key. On a Lie Detector, the player stops by itself.
- **The stuck guard** (see Safety) ends a loop that stops making progress.
- **The old `tour` stays** until the loop has passed the game bar in 3 separate sessions.

## Checking it

**The results harness comes first,** at today's code, with its own runner. The runner is built like `run_scenario`, but keeps the simulated world, which `run_scenario` throws away. It measures, up to a fixed horizon T (say 20 simulated minutes):
- EXP and kills;
- HP lost, lowest HP, potions, and the time of death (EXP stops counting there);
- piles created, and the share picked up before they vanish (from the world's drops, which know when they landed; drops landing within their lifetime of the horizon are left out);
- time without progress, from the world's log and her platform per look (`telemetry.captured()` changes nothing she does);
- how often each backstop fired (logged as telemetry events, not log lines, so the golden traces don't change).

Check the harness first: its kills must equal the world's on all 51 scenarios.

**Fair comparisons:**
- Both loops run to the same horizon; the old tour is restarted until T.
- Runs are paired on world seeds, with the policy's own random seeds fixed separately, and start platforms vary.
- The bar is per seed, for example better on at least 8 of 10 seeds and no seed more than 10% worse.
- The simulator's `SimTimeout` is a `RuntimeError`, so a loop that recovers from errors would catch it and spin. The harness stops the loop by an explicit end time instead, and `SimTimeout` becomes a `BaseException` (a simulator-only change).

**Knobs the simulator needs first** (step 0), because it hides exactly what the new rules depend on:
- hit monsters hidden for 1-3 s;
- a detection miss rate per look;
- the respawn time and the share of spawn points filled (today all 48, every 8 s);
- a competitor who kills monsters at a set rate and takes drops that aren't hers;
- a potion count;
- an option that locks the arrows during an attack (today the simulator lets her walk through one, while `Actions.walk` waits it out).

Plus the game's errors that exist already: 27.5% of stars miss, walks end a few units off, her minimap dot is read to whole pixels.

**Thinking takes no time in the simulator.** Looks already cost simulated time; what's free is the decider's computation (spot searches, route times). The decider's real computing time is measured with the real `perf_counter` and charged to the simulated clock. This knob is off by default, so the golden traces don't change.

**Golden traces:** all 51 stay a regression check until step 6, because the old tour stays behind its flag. The loop gets twins of the tour scenarios with their own traces and `expects`, one for each rule that must survive.

**Bar for a game trial:** in the simulator, with the knobs on, the loop beats the fixed old tour (step 1) on EXP a minute and on piles picked up, loses no more HP, and never dies. In game: over 3 sessions of 20 minutes, EXP a minute at least the old tour's, lowest HP above 40%, no death, and no stuck guard firing.

For scale: a 9-stop sweep in game made about 76 EXP a minute; `tour_run` in the simulator makes about 143.

## Plan

Each step is its own change, checked before the next.

0. **The results harness, the simulator's knobs** (off by default, so the golden traces don't change), **and a baseline** of today's tour over 10 seeds, with the knobs and the game's errors on.
1. **Fix the known bugs in today's tour, one change each,** so the new design isn't credited with them:
   - every plan that presses jump or attack waits out the lock;
   - the mover's walks press pick-up;
   - the 60 s loot-trip fight on its own platform;
   - kills on the trips to rest and to stand by and in guard fights are counted;
   - the report adds up HP lost.

   Each re-records its scenarios, with the diff explained. Then re-run the baseline: that's "the fixed old tour".
2. **An explicit mode per trip, with no behavior change.** It must reproduce today's three cases exactly:
   - **avoid:** no escort (`nav`);
   - **hunt:** escort with bursts (trips to spots, loot trips, and the stand-by trip, which keeps today's bursts until the loop makes it transit in step 5);
   - **transit:** escort without bursts, potion-only health at the navigator, its fights keeping the hunter's own health check (the trip to rest).

   The mode carries the kinds hunted. The new parameter is keyword-only, and its default is the navigator's own mode. Calls that keep today's behavior don't pass it, so tests that stub `nav.go(pid, x, seconds)` keep working. All golden traces stay identical.
3. **`Navigator.step()`, with no behavior change.** A step is exactly one pass of today's `_go` loop, including a whole `follow()` with its queued chains. A trip-in-progress object carries what passes keep today: the quick flag, when the last plan ended, the avoid memory, tries, fights, waits, misses, the deadline and the route weights. `Trip` keeps its fields, and `go()` still sends `trip`. A test runs every nav scenario through `go()` and through a hand-written step loop and compares the traces.
4. **The ledger and the drop store as observers,** fed by `Senses.listeners` and read by nothing that decides yet. All golden traces stay identical; the check is that ledger kills equal world kills in every scenario (`tour_run` 114, `tour_hunt_basin` 58).
   - Some tests fake gains by bumping `h.report.exp_kills` without a look (`tests/test_fight.py`). The ledger's own tests drive looks through Senses instead.
5. **The hunt loop behind `tour --loop`,** reading the ledger and the store, with the loop's twin scenarios and the `hunt` topic on the mini-map. It's compared with both baselines; iterate until it passes the bar.
   - The value-map test that pins EXP per visit (`tests/test_navigation.py`) gains a visit-time field.
6. **Game trials** with you there. After 3 passed sessions the loop becomes the default. Then the stops, loot trips and their caps are removed, with their golden traces.
7. **Transit with avoidance** for rest and standing by, behind an option, with its own bar: HP lost on rest trips, and no deaths.

## Open questions

These need a measurement in game, or your call:

- **How long do drops last, and how long are they only hers?** This drives rules 6 and 10. Measure it by watching a drop left alone in a recording.
- **The respawn time, and how many monsters are really out** with other players about. Rules 8 and 9 and the fast exit depend on it; today's 8 s is the simulator's guess.
- **Can she walk during the 730 ms attack lock?** If so, rule 4 could become throw-and-walk instead of stop, throw, go.
- **What does a look cost end to end in game?** Detection alone is measured: about 100 ms for the band, 300 ms for the whole screen. That sets how many looks fit in a throw's cooldown.
- **Should hunt mode's chains see monsters?** Today, with the fighting escort, chains are planned with no monsters at all: no danger checks, no waits, no hops. The recorded reason is "it fights them instead". With rule 4 shooting only within a budget, a monster on a later leg is caught only by the hit cut.
- **Should a full inventory ever be cleared?** Today it lasts the whole session, and items stop being looked for.
- **Several kills between two looks count as one,** since the EXP number only says it changed. Should the hunter read the number itself?
- **One map only.** Portals and other maps are out of scope.
