"""Screen geometry every layer shares: the 1920x1080 game window the templates were made at, the
half-size frames looks work on, and how map units show on screen.
"""

GAME_SIZE = (1920, 1080)     # game window size the templates were made at
SCREEN_SCALE = 1080 / 768    # screen px per map unit at that size (the game lays out a 1366x768 view)
SCALE = 2                    # looks work on frames shrunk by this much: half size
UNITS_PER_HALF_PX = SCALE / SCREEN_SCALE  # map units per half-size frame pixel
HUD_TOP = 0.88               # the bottom bar starts this far down: monsters and drops are looked for above it
