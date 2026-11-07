# Client update of 2026-11-04 (version 1.2.0): what changed

Build made 11-04 11:30 PT (exe linked 09:08 PT), on this PC from about 17:30 that day. Compared against the 10-28 build. Everything here comes from reading the files on disk; the client was never attached to or run for this.

## Summary

The update is about abuse control. It adds a GM tool for checking players on a map, plus limits on megaphones and on new accounts. In code that could be read, nothing changed in maps, monsters, skills, UI pictures, movement or combat.

## Files

10 of 478 entries changed in content; the other 367 files got new timestamps only.

| File | Change |
|---|---|
| `Data/Quest/QuestData/QuestData_000.wz` | Quest 10213: "in this stae!" became "in this state!". The only data change. |
| `MapleStory.exe` | +475,648 bytes; label "Ver. 1.1.0" became "Ver. 1.2.0" |
| Canvas, Gr2D_DX11, NameSpace, PCOM, ResMan, Shape2D, Sound, ZLZ64 DLLs | Relinked with no change. Compared byte by byte with the 10-28 versions from Nexon's patch server: the only differences are the link time and checksum in the headers, the debug directory's timestamps, the PDB ID and the signature. Code and data are identical. |

Anti-cheat files in `appdata/grap` are dated 09-09 or earlier, and its logs are encrypted.

## Exe: every change found

### 1. New GM panel, "Map Users" (`/gmpanel`)

74 new functions, about 44 KB, at RVA 0x26b4540-0x26bf8xx. It can dock in the game or pop out as its own window, in a Large or Small layout. It works only on maps with 100 users or fewer.

| Column | Meaning | Comes from |
|---|---|---|
| State | blank, Left (other map), Other ch (other channel or offline) | presence updates |
| Name, AID, CID, Lv, Job | identity | list |
| Stay | time on map | list |
| Hunt | current hunting run | list |
| Jail | jail count (decays) | list |
| Bans | login bans | details |
| Rpts | macro reporters | details |
| NX | NX of own Cash Shop buys (gifts sent not counted) | details |
| LD | Lie Detector status | LD results |
| Chase, Ban, LD buttons | actions | |

The help text is the panel's own. Clicking a cell copies it.

Server messages it handles (two new packet IDs in the admin dispatcher, RVA 0x18e5350):

- `0x1b8`, sub 0: list. Each row: CID, AID, name, level, job, stay, hunt, jail.
- `0x1b8`, sub 1: details. Each row: CID, bans, reports, NX (64-bit), plus a flag that shows "?" when the data is unavailable.
- `0x1b8`, sub 2: presence. Each row: CID, state.
- `0x1b8`, sub 3: refusal, for a map over the limit.
- `0x1b9`, sub 0: Lie Detector results per user, with a chat summary.
- `0x1b9`, sub 1: one user passed or failed.

Lie Detector status codes: 0 Not here, 1 No attack, 2 Cooldown, 3 Solving, 7 Sent, 0xFE Map blocked, 0xFF Type off, -3 Passed, -4 Failed, others Fail(n). The server assigns these. The panel's text for a bulk send: "Skipped: already Sent/Passed/Failed, other channel, no attack in 60 s, got one in 2 min, blocked map."

Ban dialog: "Ban %s - login ban, disconnects if online in this world. Longer bans: OPTool." Hours: 1, 2, 3, 6, 12, 24. Reasons 1-16: HACK, BOT, AD, HARASS, CURSE, SCAM, MISCONDUCT, SELL, NAME, TEMP, GM, IPROGRAM, MEGAPHONE, Solicitation, Bug Exploitation, Manual Temporary Block. `/block` and `/chase` already existed; the panel puts them behind buttons.

Who can open it: I found no client-side check near the command. The data only arrives in admin packets, so the server is what controls it. Not tested, and typing it on a real account would be unwise.

### 2. Chat limits skip GM characters

Two functions (0x16445f0, 0x1651810) enforce "Too much chatting can disrupt other players' ability to play the game" and "Repeating the same line over and over can negatively affect other users". Both now skip the limit when 0x16528a0 returns true: job 900-999 or 800-899, each with a further flag (GM and tester characters, by my reading). That fits the panel sending commands in bursts. Limits for normal characters are unchanged.

### 3. Character creation now counts clicks

The login stage object grew 64 bytes (656 to 720). The new member at +0x238 holds 15 numbers, capped at 65535:

- 9 click counters for the appearance arrows: gender, face, hairstyle, hair color, skin color, top, bottom, shoes, weapon. Gender and the arrow wiring are confirmed in code; the rest are matched by the window's control names.
- 1 click counter for the STR, DEX, INT and LUK arrows (0x1bdf0d0).
- 4 values: the stats as the creation screen sets them up, stored when the counters reset (0x1be0340).
- 1 counter for names the client's own filter rejects on pressing create (0x1bcae20).

Functions to write all 15 into a network packet and read them back, as 16-bit values, were added (0x48c690, 0x48c730). No readable code calls them. They survived the linker, so hidden code probably does: 0x1bcf330 is fully hidden and grew 43 bytes. Sending is therefore my inference, and the client doesn't state the purpose: it could be bot detection for mass-created accounts, or plain analytics. It only runs in the creation screen.

### 4. New and changed client messages

Decoded from the exe's message table, the same way the client decodes it, across all five language tables. The table went from 6,293 to 6,295 entries, and these three are the only differences:

- New 6293: "You cannot use megaphones for a limited time."
- New 6294: "A new game account can't be created with this email domain."
- Changed 6216: "If you buy this item, you will not be able to sell or gift cash items to other players **or use megaphones** for %d hour(s). Do you want to continue?"

### 5. Megaphone limit

0x27a470, called from the item-use code at 0x2d8db30, blocks six item types (0xC-0xF, 0x26, 0x4F) for a set time after a stored date, using a server setting (setting 0x19; 0 means 4, negative means off), and shows message 6293. The message and the changed purchase warning are why I read these as megaphones, hours and the purchase date.

### 6. New login result

Login result `0x92` (0x1bc8b40) shows message 6294. The same message was added to the result-to-text function at 0x1be5920.

### 7. Smaller items

- Window message handler (0x2d4ec90): one mouse-click path now also runs for a second window state. It fits the panel's pop-out window (my reading).
- Panel hooks in two existing functions (0x2da1b40, 0x2da9560), and the command in 0x20bbb00 and 0x2dad620.
- String removed: `mapleIDOnlyContainLettersAndNumbers`. No readable code refers to it.
- All other differences are source line numbers and shifted offsets.

## What "macro reporters" means

The panel's `Rpts` column is a plain number per character, sent by the server to GMs along with Bans and NX. The client has no other definition of it. In the client's wording a macro is a bot ("macro-assisted program").

Players have two ways to flag someone:

- **Report Player** (`UI/Report.img`, the right-click menu's `BtClaim`, a report button on incoming messages). Reasons include "Illegal Program Use", described as "Suspected use of third-party program". There is a daily limit ("Daily Report Count: %d/%d"), the target must be on the same channel, and "False reports will result in penalties".
- **Lie Detector Test item** (2190000-2190002, right-click menu's `BtLieDetector`). "Can be used on suspected botters. Usable only when a suspect is hunting." The client also has the message "%s has used the Lie Detector Test."

My reading: `Rpts` counts players who reported that character for botting. The client doesn't show which of the two feeds it, or over what period.

A player can't see their own number. The only sign is the popup "You have been reported by a user." (result code 3 in 0x2e53ce0), which the server has to trigger. The project's logs don't capture popups, so its absence there says nothing.

Already in the client before this update (not new): "The Lie Detector Test confirms that you have been botting. Repeated failure of the test will result in game restrictions.", "You have been sanctioned as a macro-assisted program user.", and the GM monitoring notice (message 2200).

## What "Player lie detector - key pattern type" is

Also not new; it's in the 10-28 build too. It's one value of the log field `hacklogactiontype` (named at 0x5bb380, table entry 254), which names the actions in the game's hack-log system. The values:

No functionality; Lie Detector; For devs to check (tracking video); Send Email; For admin team to check (tracking video); Kick + count daily, then protection mode; Protection mode; Disconnect; Violetta; Return True; Click, Quiz and String Lie Detector; Save to record; Client force-crash and protection mode; Send Email 2; Video filming - hack log pattern; Video filming - macro hack log pattern; Collect process information; Take screenshot; Film video + take screenshot + collect process + packet dump; Designated a macro for 3 hours. (twice); Player lie detector - key pattern type; Client force-crash + protection mode; Account Blacklist (Random player lie detector); Client Reinstall Guide and Exit; HackLog Instant Ban (1 Day, 3 Days, 7 Days, Indefinite); Suspected Fraudulent Transaction Restriction (7-Days).

The client doesn't define "key pattern" and holds no rule or threshold under that name. My reading: a keyboard-input pattern flagged by the hack-log system, answered with a Lie Detector. The key fingerprint described below is the closest match in the client, but nothing in the client links the two. That's inference.

## Reason labels

Function 0x282ba0 turns a number into a label. No readable code calls it and no table points to it, so its caller is protected code, or it's unused.

| # | Label | # | Label |
|---|---|---|---|
| 0 | Field Mob Hunting | 9 | Hack Log_T |
| 1 | Keyboard Macro | 10 | NGS |
| 2 | Mob Herding | 11 | Hack Log Pattern |
| 3 | Macro Log | 12 | LiveActionSystem |
| 4 | Hack Log | 13 | LiveActionSystem_Top100 |
| 5 | Pet Teleport | 14 | Report Player |
| 6 | (none: "No Reason String %d") | 15 | Hack Log_Analysis |
| 7 | Auto Hotkey, Virtual Keyboard | 16 | Boss |
| 8 | Macro Log_T | | |

NGS is the anti-cheat's name. "_T", "LiveActionSystem" and "Top100" aren't explained anywhere in the client.

The messages stored next to them are "World: %02d", "Channel: %02d", "Detection: %s", "Enter Channel: %s", "Recording: %s", "Field: %09d" and "Waiting: %d", followed by Trade_Count, Trade_Meso, Harvesting_Count, Harvesting_Time, Hunting_Time, Chat_Length, Chat_Contents and "No trigger". The log class `ETC_UserReplayData` has a `recordreason` field, and hack-log actions include "Video filming - hack log pattern".

My reading: a staff or recording tool that lists players with the reason each was picked. That's inference. I found no readable code that uses these messages.

## The input log, in detail

None of this appears to be new on 11-04. Of 20 functions compared with the 10-28 build, 17 are identical with addresses masked. The key handler and key recorder contain protected blocks, so they can't be fully compared, and the reason-label function differs in its masked hash at the same size. Exact cut-offs are left out of this section on purpose: it says what's collected and when, not where the limits sit.

**Correction.** I earlier gave "UI Event, Keyboard Scan Code, Chat Window Creation" as the values of `inputeventtype`. They're the labels of a different field, `devlogtype`.

### The record

`ETC_MSInputEvent`, built by 0x572410:

| Field | Holds |
|---|---|
| `sn` | A counter for each stream |
| `charid` | The character |
| `inputeventtype` | 0, 1 or 2, as below. The client has no names for them |
| `datasize` | Bytes or entries in this batch |
| `deviceresolution` | Despite its name, the input device's ID number |
| `key_vid`, `key_pid` | The device's vendor and product IDs. 0x1aed280 reads `VID_` and `PID_` out of a `#`-separated device path |
| `keydatas` | Key records (types 0 and 2) |
| `mousedatas` | Mouse records (type 1) |

Each batch also sends the game server a summary: packet 0x1ef, kind 0x27, with the counter, type and size. The records themselves go to the client's logger. I didn't trace where the logger sends them.

### Type 0: running key log, per device

The window-message handler 0x2d4db10 passes key messages to 0x1ae4c70, which hands each one to a recorder object for its device (0x1af2590). A record is text, either of:

`time,key,message,value,0,,`
`time,key,message,value,ID,x,y`

- `key` is a Windows virtual-key code, and `message` is the key-down or key-up message.
- `value` is looked up from the key and the character (0x1429f0050). Probably the key's game binding; inferred.
- `ID,x,y` come from an object in the same code family as the local character, when a check passes. Probably her position; inferred.

Batches go out about every 5 minutes, sooner if they grow large.

### Type 1: mouse records, not confirmed active

The format is `time,message,x,y,value`. The function that formats them (0x1b14c60) throws the text away, and the function that would store it (0x1b14de0) has no readable caller. Neither does the flush (0x2dd39e0). Either this route is dead or protected code drives it.

### Type 2: key fingerprint, started by the server

1. Server packet 0x5ac starts a capture (0x1b00f90).
2. The client stores each device's key releases with their times (0x1b01950).
3. After 1 minute, it picks each device's rarest key as a starting point (0x1b01c30).
4. After 6 minutes, it takes a fixed-length run of keys from that point, if the keys are varied enough, and computes an MD5 hash and a small XOR value (0x1b02050). Packet 0x1f3, kind 4, sends these pairs to the server.
5. Server packet 0x5ad carries a hash. The client answers with the raw key list and the text `HashValue|<hash>|Time|<hhmmss>`, as a type 2 log (0x1b01140).

My reading: the same key sequence gives the same hash, so the server can spot a sequence that repeats across sessions or accounts, and ask for the raw keys behind a hash. The client side is confirmed. The purpose is inference.

### Related reports outside this log

- **Device list** (packet 0x1ef, kind 100): every 30 minutes if more than one input device has been seen. Each device's ID, vendor and product IDs, and more data I didn't decode.
- **Held arrow key** (packet 0x1ef, kind 0x60): an arrow key is down and no arrow has been released for a long time.
- **Repeated clicks** (packet 0x2a2, value 1): the client keeps a history of rounded click places and timing, and flags it when a random slice of that history appears again.
- **Cursor movement** (packet 0x2a2, value 0): a check on the shape of fast cursor moves that lead to a click.
- **Cursor path** (packet 0xaf): recent cursor and click positions, sent from six functions when a dialog ends. Five of them sit next to the Lie Detector UI code, but I didn't confirm which dialogs they are.
- **`ETC_MSMouseEvent`** (from 0x2ef2f90): event count, two click hashes, screen size. A server setting can switch it off.

### Not determined

- How often the server starts a capture, or for whom.
- What the server or staff do with any of this. No rule or score is in the client.
- Whether protected code adds to these logs or reads them.

## What couldn't be checked

- 703 functions contain blocks the protector replaced with a jump and random filler (294 KB). The real code is in the encrypted `.boot` section at runtime, so those blocks can't be compared on disk. Some are in combat code: the functions mentioning `shoot`, `RemoteMelee`, `RemoteForceAtom` and cooldown logging. A logic change there can't be ruled out. Their readable parts match.
- `.boot` grew 414 KB and `.themida` 205 KB (virtual). Weak evidence either way: the exe's total size has swung by 110-840 KB in every recent update.
- Comparing the protected code would need a memory dump of the running client, with an anti-cheat driver loaded. I didn't do that.

## How it was checked

- Old exe recovered from the 11-03 Ghidra project (MD5 and size match the 10-28 build's manifest).
- 123,563 and 123,680 functions from `.pdata`, normalized with addresses masked: 122,339 identical.
- The rest compared by following control flow, reading switch tables as tables, then by decompiled text with addresses masked. A plain linear comparison flagged 908 false differences.
- Floating-point constants read by 115,763 paired functions: 0 differences.
- The 63 functions the Violetta study decompiled: all unchanged, addresses moved.
- Old DLLs: `download2.nexon.net/Game/nxl/games/59813/59813/<first 2 of hash>/<hash>`, the 10-28 manifest's chunks, each zlib-compressed and matching its SHA-1 once unpacked. ZLZ64's MD5 also matches what Ghidra logged from the install on 11-03.
- Scripts, raw results and the old exe and DLLs: `D:\MapleHelper-scratch\client-update-1106`.

## Game data refresh, 11-06

- `gamedata/data/copy`: all 270 files match the install by content. The last two were swapped at 08:08, with perception, pet and the web UI stopped for it (the user's OK); all three are back.
- `gamedata/data/out/tables`: rebuilt, and identical to a build from the final copy. The old tables were from 10-23, older than the copy: the rebuild added 19 monsters, 30 maps and new skill columns. Entries for Tree Dungeon and the other current maps are unchanged, as are the thief's and first-job magician's attack numbers.
- Map art caches redrawn: identical to before.
- 292 table-related tests passed, before and after the last swap.
- `gamelab.sim.baseline.data_stamp` changed, so cached harness baselines refill on their next run.
- Old versions of the swapped files: `D:\MapleHelper-scratch\client-update-1106\copy_build_1022`.
- Next refresh: `refresh_copy.py` in the scratch folder. `python -m gamedata copy` writes in place, so don't use it while the services run.
