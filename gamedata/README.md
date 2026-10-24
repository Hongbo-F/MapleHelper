# gamedata

Read the game's own data files: item and monster names, monster stats, maps, skills, movement physics and animation timing. MapleHelper's item names, EXP and combo timing can then come from the game itself instead of fan sites or guesses.

It only reads a **copy** of the game's `appdata\Data` folder. The game folder is read once, to copy it, and never written. The copy and everything extracted from it stay in `gamedata/data/`, which git ignores.

## Use

From the repo root, in the `maplehelper` conda env:

```bash
python -m gamedata copy --install D:\maplestorycw
python -m gamedata tables
```

1. `copy` copies `appdata\Data` (about 510 MB without Sound) into `gamedata/data/copy`. Run it again after a game patch: unchanged files are skipped.
2. `tables` writes JSON tables to `gamedata/data/out/tables`:

| Table | Holds |
|---|---|
| `items` | Etc, Use and Equip names by item ID |
| `mobs` | Every monster: name, level, HP, EXP per kill, speed, attack |
| `maps` | Every map: names, town or not, bounds, monsters spawned, exits |
| `skills` | Every skill: name, animation, MP cost, damage, duration per level |
| `physics` | Movement constants: gravity, jump speed, walk speed |
| `actions` | Frame delays (ms) of each body animation: attacks, jump, walk |

To look around:

| Command | Shows |
|---|---|
| `list` | Every package in the copy, with its image count |
| `list String/String_000.wz` | The images in one package |
| `dump Packs/Skill_00000.ms Skill/400.img --depth 4` | One image's property tree |
| `export String/String_000.wz` | Every image of a package as JSON, in `gamedata/data/out` |
| `pictures UI/UI_000.wz LieDetector.img` | Every picture of an image as PNG, at its path in the image, with the tree as `tree.json` (each picture naming its file), in `gamedata/data/out/pictures` |

## The formats

- **`.wz` packages** (`wz.py`): the classic Wizet "PKG1" format. Each category is split into `X.wz` (an index), `X_000.wz` (the data) and `_Canvas/` (pictures). Text is XOR-masked, not encrypted, so no key is needed. Directory offsets are scrambled with a hash of the client version, found by trial.
- **`.ms` packs** (`ms.py`): newer clients keep Mob and Skill data in `Data\Packs\*.ms`, encrypted with ChaCha20 under keys derived from the file name and a stored salt. Only the first 1 KB of each image is encrypted; inside is an ordinary .wz image. Follows WzComparerR2's reader (MIT).

## Limits

- **Not in the data files:** chat and pickup message templates, the EXP table, damage and attack-speed formulas, and drop tables. Those live in the game program or on the server.
- **Pictures:** `canvas.pixels()` decodes BGRA4444, BGRA8888 and RGB565, which covers monster sprites. DXT-compressed pictures aren't decoded yet. `canvas.mob_frames()` collects a monster's frames: the frame data is in `Packs/Mob_00000.ms`, the pixels at the same path in `Mob/_Canvas/_Canvas_000.wz`, and repeated frames point to others with `_inlink`. `canvas.minimap_markers()` gives the minimap's markers (hers, other players', NPCs', portals'): the pictures are in `Map/_Canvas/_Canvas_000.wz`, the origins in `Map/Map_000.wz`, both under `MapHelper.img/minimap`. `canvas.status_numbers()` gives the bottom bar's number glyphs (`UI.wz` `StatusBar.img/gauge/number`: the digits, comma, point, percent and brackets of "13,549[67.02%]"), held in `UI/UI_000.wz` itself but for `%`, which is in `UI/_Canvas/_Canvas_000.wz`.
- **UI pictures** are 1x1 placeholders in `UI/UI_000.wz` whose `_outlink` names where the pixels are, from the copy's root ("UI/_Canvas/Basic.img/BtOK2/normal/0", sometimes another image). `canvas.source()` follows those links and `_inlink`s; `pictures` writes them out. The game draws them at 1366x768 and scales the screen up, so on a 1920x1080 screen they're 1.41 times their size.
- **The Lie Detector's windows** (the game's anti-bot check; gamelab's `perception.lie_detector` spots them) are in `UI.wz`, three games of it:
  - `AntiMacro.img/lieDetectorStandBy` and `lieDetector`: "find the Violetta wearing make-up", four Violettas under spotlights, picked with buttons 1 to 4. Its timings are in `lieDetector/info` (15 s to wait, 2.5 s of shuffling).
  - `LieDetector.img/figureTrackingLieDetector`: the mouse mini-game, following a fading figure with the cursor.
  - `AntiMacro.img/soundLieDetector`: four numbers read aloud and typed on a keypad. Not seen in game yet.
- **Sounds and the embedded font** are listed but not decoded.
- **IDs differ from the old game** for monsters, skills and maps (Snail is mob 2, Lucky Seven is skill 4001003). Item IDs are the old ones.
- **Only .ms version 4** is supported. A patch that changes the format makes `MsPack` refuse the file rather than misread it.
