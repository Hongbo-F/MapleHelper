# Runtime system map: data format

The architecture page's first view answers four questions about the running system, from the code:

1. **Entry points**: what gets launched, by whom (the user, the babysitter, a UAC click, another process), with what
   command, and how it ends.
2. **Start-up and communication**: who spawns whom, and every channel two processes talk over.
3. **Who reads from the game**: which processes take pixels or window state from the game (or its files), how, how
   often, and for what.
4. **Who sends input to the game**: which process finally calls SendInput (or moves focus, the pointer, the window),
   and the path a key takes from a decision to the game.

Files live in `tools/architecture/data/system/`, one per topic. Check one with
`python tools/architecture/validate_system.py tools/architecture/data/system/<id>.json` (must say OK; fix warnings).

Rules: every claim about the code carries `evidence`: `[{ "path": "gamelab/control/player.py", "line": 1023,
"symbol": "Player._press" }]` (repo-relative path, the real 1-based line, the symbol defined or used there; `symbol`
may be "" for a plain statement). Never invent; the code wins over README prose. Plain English for an engineer new to
the codebase; Markdown in `*_md` fields. Process ids: use these, and add a new one to entrypoints.json only if none fits:
`player` (the elevated key player), `perceive` (the perception service), `hunt` (any hunting command: hunt, tour,
grind, nav, probe, patrol...), `pet` (`pet --every`, the pet feeder and its guard), `park` (`nav --park`), `minimap`
(the mini-map window and/or `web`), `lie-tracker` (the service's mini-game tracker child), `partwriter` (the
recording's part writer child), `recorder` (the old `gamelab recorder`), `tools` (short commands: stop, clear-alarm,
snap, spot, keys, ...), `helper` (`python -m maplehelper`, the original combo app), plus the outside ids `game`,
`user`, `windows`, `babysitter`, `phone`, `cloudflare`, `claude`.

```jsonc
// entrypoints.json
{ "id": "entrypoints", "summary_md": "...",
  "processes": [ { "id": "perceive", "name": "The perception service", "kind": "service",   // service | command | child | helper-app | external
      "command": "python -m gamelab perceive", "started_by": ["babysitter", "user"],
      "how_started": "how exactly (terminal, WMI Win32_Process.Create hidden, ShellExecute runas = UAC, subprocess.Popen with pipes, ...)",
      "elevated": false, "single_instance": "the mutex or lock that keeps it to one, or ''",
      "lifetime": "how long it lives", "stops": "every way it ends",
      "reads_game": true, "sends_input": false,
      "summary": "2-4 sentences", "details_md": "optional",
      "evidence": [ { "path": "gamelab/__main__.py", "line": 2649, "symbol": "cmd_perceive" } ] } ],
  "spawns": [ { "parent": "perceive", "child": "lie-tracker", "how": "subprocess.Popen, stdin/stdout pipes",
      "when": "while the Lie Detector alarm is up", "evidence": [...] } ] }

// channels.json: every way two processes (or a process and the game/Windows/the user) exchange data or signals
{ "id": "channels", "summary_md": "...",
  "channels": [ { "id": "service-pipe", "from": "perceive", "to": "hunt", "direction": "two-way",
      "transport": "named pipe",      // named pipe | shared memory | file | stdin/stdout | telemetry file | mutex | process list | push | ...
      "name": "\\\\.\\pipe\\gamelab-perceive", "carries": "what goes each way", "format": "pickled dicts, ...",
      "rate": "every frame, 15/s", "lifecycle": "who creates/removes it", "details_md": "optional", "evidence": [...] } ] }

// readers.json
{ "id": "readers", "summary_md": "...",
  "readers": [ { "process": "perceive", "what": "pixels",   // pixels | window state | game files | other
      "how": "DXGI desktop duplication (capture.Grabber 'dxgi'), the game's client area", "rate": "15 frames/s",
      "purpose": "...", "when": "always | while a plan runs | on demand | only with --no-feed | ...", "evidence": [...] } ] }

// input.json
{ "id": "input", "summary_md": "...",
  "senders": [ { "process": "player", "api": "SendInput (keyboard, scan codes)", "what": "...",
      "conditions": "the guards before each key", "evidence": [...] } ],       // also focus changes, pointer moves, window resizes
  "plan_sources": [ { "process": "hunt", "module": "gamelab/control/actions.py", "symbol": "Actions.play", "kind": "plan kind or ''",
      "what": "which plans it makes", "evidence": [...] } ],
  "path_steps": [ { "process": "hunt", "path": "gamelab/...", "symbol": "...", "line": 1, "text": "one step of a key's life, in order" } ] }

// sequences.json: start-to-finish stories across processes, for sequence diagrams
{ "id": "sequences", "summary_md": "...",
  "sequences": [ { "id": "cold-start", "title": "...", "summary": "...",
      "steps": [ { "from": "babysitter", "to": "perceive", "channel": "channel id or ''", "what": "1-2 sentences",
                   "evidence": [...] } ] } ] }
```

## components.json: what runs inside each process

The page's main view is written for a person: for every process, in plain words, what it is, whether it is its own OS
process, who is expected to start it (you by hand, the babysitter automatically, another process) and when, how long
it lives, and the components inside it: its main loop, each thread, each child process it starts, each in-process
subsystem worth knowing (the player's watchdog, the service's frame loop and its per-client pipe writers, the hunt's
Hunter/Navigator loop and its feed client, the telemetry writer thread, the guard inside the pet feeder...).

```jsonc
{ "id": "components", "summary_md": "...",
  "processes": [ { "process": "player",            // a process id from entrypoints.json / SYSTEM.md
      "is_os_process": true,                       // false only for something that is never its own process
      "plain_md": "4-8 sentences a person reads first: what it is; standalone OS process or not, elevated or not; who is expected to start it, how and when (by hand in a terminal, the babysitter automatically, spawned by X); how long it lives and what ends it; what it does with the game.",
      "expected_start": "you, by hand | the babysitter, automatically | spawned by <process> | ...",
      "components": [ { "id": "watchdog", "name": "The Lie Detector watchdog",
          "runs_as": "thread",                     // main thread | thread | child process | thread pool | timer | callback | library code on the caller's thread
          "started_by": "Player.serve at start-up", "lives": "as long as the process",
          "does": "1-3 sentences", "rate": "4 times a second",
          "talks_to": [ { "to": "plan runner", "via": "threading.Event Alarm", "what": "stop now" } ],
          "evidence": [ { "path": "gamelab/control/player.py", "line": 1, "symbol": "Watchdog" } ] } ] } ] }
```
