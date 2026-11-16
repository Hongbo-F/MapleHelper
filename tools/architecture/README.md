# The architecture page

One self-contained web page that maps gamelab and maplehelper: what runs on the PC while she plays (which processes,
who starts each, what they say to each other, who reads the game and who presses its keys), step-by-step scenarios,
and the code layer by layer down to each module.

- **See it:** `/architecture/` on the web UI (`python -m gamelab web`; through the tunnel it takes the Cloudflare
  Access login, as `/tls/`: `gamelab/ui/architecture.py`).
- **Build it:** `python tools/architecture/build.py` writes `gamelab/lab/architecture/index.html` (outside git), which
  the web UI reads at each request. It joins the hand-written data in `data/` with what the code says itself (imports,
  line counts, which tests import what, the layer rules of `tests/test_architecture.py`).
- **Check the data:** `python tools/architecture/validate.py --all` (the layer slices, `SCHEMA.md`) and
  `python tools/architecture/validate_system.py --all` (the running system, `SYSTEM.md`). Each claim cites the code by
  path, line and symbol; after an edit moves code, the warnings name the citations to move.
- **The page:** `page.html` (the template), `system_intro.md` (the system view's opening words).
