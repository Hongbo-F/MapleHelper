"""Control: decisions turned into key plans for the elevated key player, and the closed loops
that carry out moves and routes. Also the player itself and the Lie Detector alarm.

  actions    her actions (throws, walking, picking up, potions, the chair) as key plans
  mover      one route move at a time, with a look before and after
  navigator  getting to a platform: plan the route, chain its moves into one plan, replan
  client     handing plans to the player and getting the results (files in gamelab/lab)
  player     the key player itself, run as administrator
  alarm      the Lie Detector alarm: refusing to run, and paging whoever is at the PC
  timeline   MapleHelper's own combo code as key timelines; her setup from config.toml

May import: perception, localization, motion, and the shared modules. The player (player.py)
runs as administrator and loads its code once: keep what it imports small.
"""
