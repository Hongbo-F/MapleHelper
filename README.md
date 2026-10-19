# MapleHelper

One-button escape combos for a dart thief, with an on-screen coach that narrates every step so she can learn the moves.

| Button | Combo | What it does |
|---|---|---|
| F1 | Jump-Throw | Jump, then attack in mid-air so a star is thrown (no claw swipe) |
| F2 | Back Off & Throw | Walk away, turn around, throw twice |
| F3 | Hop Back & Throw | A Jump-Back Shot, then one more throw |
| F4 | Jump-Back Shot | Jump away, turn around in the air and attack; the star comes out just as she lands |
| Esc | | Stop the running combo and cancel any queued one |
| F12 | | Slow / demo mode: pauses before each step so she can read the coach |

"Away" means away from where she's facing when she presses the button. The helper tracks facing from whichever turned her last: her own ← / → press, or a combo's.

A combo button pressed in the last 300 ms of a running combo (`buffer_window`) is queued to start the moment it ends. If she presses several, only the newest is kept. Pressed earlier than that, it's ignored, so mashing at the start of a move does nothing. Pressing F4 again as she lands chains Jump-Back Shots.

When a combo ends, the coach shows the whole recipe, e.g. `Hold ← to walk away · Tap → to turn around · SHIFT (Lucky Seven) ×2`.

## Setup

Pure standard-library Python (ctypes + tkinter), Windows only.

```bash
conda env create -f environment.yml
conda activate maplehelper
python -m maplehelper
```

Quit with Ctrl+C in that terminal.

## First run checklist

1. **Match her key config.** Edit `[keys]` in [config.toml](config.toml) to match her in-game Key Config. If she hasn't learned Lucky Seven yet, set it to `""`. Without Lucky Seven, stars are thrown with jump + attack.
2. **Check the window title.** Run `python -m maplehelper --debug`, press a key in the game, and look at the logged `focused window` title. `game_window_title` must be part of it.
3. **Play in windowed or borderless mode.** The coach overlay can't draw over exclusive fullscreen.
4. **If keys don't reach the game,** start the terminal as Administrator. Windows blocks input from normal programs to games that run elevated.
5. **Tune `[timing]` only if her gear changes.** The defaults were measured in game with her claw's regular attack. A jump lasts about 530 ms. The star comes out about 530 ms after the attack key. The game ignores jump and attack presses until about 730 ms after it. So `key_press + throw_gap` must stay above 730. Jump-Back Shot holds the away arrow from `run_up` ms before the jump until `air_delay` ms after it. It then holds the turn arrow `turn_lead` ms before the attack and keeps it held through the attack. Quick taps there were sometimes missed by the game: about 6% of shots jumped straight up or threw the wrong way. The held pattern went 160 for 160 in testing. A faster claw or a speed buff shortens these times. `walk_away` sets how far Back Off walks.

## Trying it without the game

```bash
python -m maplehelper.keyviewer
```

This opens a stand-in window titled "MapleStory Key Viewer". Focus it, press F1–F4, and it lists every key the combo sent, with timings.

## Testing in the real game

[gamelab](gamelab/README.md) plays combos in the actual game, records it, and checks every jump and throw. Use it after changing a combo or `[timing]`:

```bash
python -m gamelab combo jump_back_shot --runs 4 --chain 8
```

## Tests

```bash
python -m pytest
```

## Adding a combo

Write a function in [maplehelper/combos.py](maplehelper/combos.py), add it to `COMBOS`, and give it a button in `[hotkeys]`. A combo is a list of steps:

```python
def my_combo(c):
    c.say(f"Hold {c.back.label} to walk away")  # new step on the coach (slow mode pauses here)
    c.tap(c.back, 400)                           # hold the away-arrow for 400 ms
    c.say(f"{c.throw_label} ×3")
    c.throw(3)
```

`c.front` / `c.back` are the arrows toward / away from the monsters. `c.keys` holds her bindings and `c.t` the timings. Also available: `c.down`, `c.up`, `c.wait`, `c.jump_attack`, `c.air_throw`. Use `c.say(text, pause=False)` for steps that happen mid-air, so slow mode doesn't pause there.

## How it works

| File | Role |
|---|---|
| `hook.py` | System-wide low-level keyboard hook. Hides combo buttons from the game and skips keys the helper sent itself |
| `controller.py` | Decides what each real key press means (combo, stop, slow mode, facing, or pass through) |
| `engine.py` | Runs one combo at a time on a worker thread. Handles abort, slow mode, and releasing held keys |
| `win32.py` | ctypes declarations: `SendInput` by scan code (what games read), window title, overlay styles |
| `coach.py` | Click-through, always-on-top tkinter overlay |
| `config.py` | Loads and validates `config.toml` with readable errors |

## Known limits

- Tested only against the key viewer, not against MapleStory itself.
- If the game's anti-cheat ignores injected input, the combos won't do anything in game. The step lists in `combos.py` can still be typed into a keyboard's own macro software.
