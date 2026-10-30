"""The one rate perception looks at the game: HZ frames a second, everything once a frame (the user, 2026-10-30: "make
perception output the 15fps whole screen only, no need for a separate 30fps existing tracking", and of her place, the
HUD and her touches: "I meant everything at 15 fps"). Every frame gets one pass of the perception service's world step
(localization.worldstate.WorldStep.step: her look and the whole screen's search at once, her place, vitals and touches,
the monsters matched to their tracks, their bars and deaths, the drops), so what the stack reads of the game, her place
and HP as much as the monsters, is never more than a frame and a pass old.

Whatever counts frames takes its rate from here (localization.feed.HZ, service.HZ, senses' glances, control.recording.FPS,
the simulator's feed): change it in this one place. What used to be counted in frames at 30 a second is counted in time
now (tracking.MIN_SEEN_S, worldstate.BAR_EVERY_S, perception.deaths.EVERY_S); the touch rules were measured on recordings
at 15 a second (perception.touches), and the locator's counts are the prototype's at 15 (localization.locate: k 1).
"""

# Frames a second (the game shows 60). 30 from 2026-10-26 (the frame loop's step with the monsters followed re-found in
# every frame, the whole screen on a thread of its own every other frame: README, Perception at one pace) until
# 2026-10-30; 15 since, one pass a frame: on the crowd's replay (D:/MapleHelper-scratch/replan/p15/cpu15.py, 900 frames,
# Idle priority on CPUs 12-23) the pass took 0.50 cores at the live rate where the step at 30 took at least 0.75, every
# state with its own frame's whole screen where the newest whole screen was 70 ms old (median) live at 30 (README,
# Perception at 15 frames a second).
HZ = 15.0
