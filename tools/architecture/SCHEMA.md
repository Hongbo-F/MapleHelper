# Architecture map: data format

The architecture page (`tools/architecture/build.py` → one self-contained HTML page) is built from the JSON files in
`tools/architecture/data/`. One file per slice of the system. Check a file with:

    python tools/architecture/validate.py tools/architecture/data/<id>.json

It must print `OK` (warnings are listed; fix every one you can: most mean a wrong line number or a name not found).

## Audience and voice

A capable engineer who has never seen this codebase and wants to understand how the autonomous player works, top
down. Plain English, short sentences, every piece of jargon explained the first time (or put in the glossary).
Concrete over vague: name the class, the threshold, the file, the number ("every 25 ms", "0.8 match score"), and
say *why* when the code or `gamelab/README.md` says why. The codebase calls the in-game character "she"; you may
too, or say "the character".

Never invent. Every name, line number, number and behaviour must come from the code or the READMEs you read. When
unsure, leave it out or say it's unclear. Line numbers are 1-based and point at the `def`/`class`/assignment line.

Markdown fields (`*_md`) may use: paragraphs, `-` lists, `**bold**`, `` `code` ``, fenced code blocks, and tables.
No HTML. No headings above `###`.

## Slice file (`data/<id>.json`)

```jsonc
{
  "id": "perception-eyes",            // the file's name without .json
  "layer": "perception",              // foundation | perception | localization | motion | attack | control |
                                      // behavior | sim | ui | analysis | shared | runtime | tests
  "title": "Perception: the eyes and the HUD",
  "tagline": "One line, at most ~15 words",
  "summary": "2-4 sentences: what this slice does and how it fits the whole",
  "details_md": "200-900 words of Markdown: how the parts fit together, the main algorithms, the important design decisions and limits",
  "groups": [                         // 2-6 sub-areas a reader can open one at a time; every module in exactly one group
    { "id": "eyes-core", "title": "Taking a look", "summary": "1-2 sentences", "modules": ["gamelab/perception/eyes.py"] }
  ],
  "modules": [ /* MODULE, one per source file of the slice */ ],
  "data_types": [                     // the important objects passed between modules (dataclasses, NamedTuples, dict records)
    { "name": "Observation", "module": "gamelab/perception/eyes.py", "line": 40,
      "summary": "what it represents and who makes/consumes it",
      "fields": [ { "name": "tag", "type": "NameTag | None", "meaning": "her name tag, if found" } ] }
  ],
  "flows": [ /* FLOW: 1-4 step-by-step walkthroughs inside this slice */ ],
  "not_done": [ "known limits, things not tried live, open questions (from code comments and the README)" ]
}
```

### MODULE

```jsonc
{
  "path": "gamelab/perception/eyes.py",   // repo-relative, forward slashes
  "title": "Eyes",                        // a human name for it
  "tagline": "One line, at most ~15 words",
  "summary": "2-4 sentences",
  "details_md": "100-700 words of Markdown: how it works inside, the algorithms, the numbers that matter, why it's built so",
  "lines": 590,                           // the file's line count
  "key_items": [                          // 3-15 of the most important classes/functions/constants, most important first
    { "name": "Eyes.look",                // Class, function, Class.method, or CONSTANT, exactly as in the code
      "kind": "class",                    // class | function | method | constant | dataclass | enum | protocol
      "line": 123,
      "signature": "def look(self, *, sweep=None) -> Observation",
      "summary": "1-2 sentences",
      "details_md": "optional: deeper notes" }
  ],
  "inputs":  [ "plain phrases: what it consumes and from where" ],
  "outputs": [ "what it produces, and for whom" ],
  "uses":    [ "gamelab/localization/pose.py" ],   // repo-relative paths of internal modules it imports/calls (gamelab, maplehelper, gamedata, tools)
  "files_io": [ { "path": "lab/inbox/<plan id>.json", "mode": "write", "what": "key plans for the player" } ],  // mode: read | write | read-write
  "concurrency": "threads, child processes, timers it starts or runs on; empty string if none",
  "tunables": [ { "name": "POLL_S", "value": "0.025", "meaning": "how often telemetry is read" } ],   // the constants that shape behaviour
  "decisions": [ { "date": "2026-10-24", "what": "the change", "why": "the reason given" } ],          // dated design decisions, from the README/comments; date "" if unknown
  "gotchas": [ "non-obvious behaviour or pitfalls" ],
  "tests": [ "tests/test_eyes.py" ]       // test files that exercise it (grep tests/ for the module name)
}
```

### FLOW

```jsonc
{
  "id": "look-to-keys",
  "title": "How a look becomes keys",
  "summary": "1-3 sentences",
  "trigger": "what starts it",
  "steps": [
    { "module": "gamelab/perception/eyes.py", "symbol": "Eyes.look", "text": "what happens here, 1-3 sentences", "process": "" }
  ]
}
```

## Special files

- `runtime.json` (layer `runtime`) also has `processes`, `lab_files` and `cli` (see below), plus the usual fields for
  `gamelab/__main__.py`.
- `flows.json`: `{ "id": "flows", "flows": [FLOW...] }` — end-to-end walkthroughs across slices; each FLOW may add
  `"slices": ["perception-eyes", ...]`.
- `glossary.json`: `{ "id": "glossary", "glossary": [{ "term", "meaning", "see": [paths] }], "timeline": [{ "date", "title", "what", "see": [paths] }] }`.
- `tests.json` (layer `tests`) may add `"test_index": [{ "path": "tests/test_x.py", "covers": [paths], "summary": "one line", "slow": false }]`.

```jsonc
// runtime.json extras
"processes": [ { "id": "player", "name": "The key player", "command": "python -m gamelab player --elevate",
                 "elevated": true, "lifetime": "how long it lives, who starts/stops it",
                 "summary": "2-4 sentences", "details_md": "...", "modules": ["gamelab/control/player.py"],
                 "threads": ["the watchdog: checks the screen 4x a second for the Lie Detector"],
                 "talks_to": [ { "process": "hunt", "via": "files in lab/inbox and lab/outbox", "what": "key plans in, results out" } ] } ],
"lab_files": [ { "path": "lab/inbox/", "format": "JSON plan per file", "written_by": ["gamelab/control/client.py"],
                 "read_by": ["gamelab/control/player.py"], "purpose": "..." } ],
"cli": [ { "command": "hunt kill", "summary": "one line", "entry": "gamelab/__main__.py", "symbol": "cmd_hunt",
           "line": 1234, "sends_keys": true, "details_md": "optional" } ]
```
