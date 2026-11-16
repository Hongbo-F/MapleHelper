"""The architecture map in the web UI (the user, 2026-11-16: "Can you serve the page on the webui and I will take a look"):
/architecture/ serves the page tools/architecture/build.py writes (paths.LAB/architecture/index.html), one
self-contained HTML file. Build it again after the code or tools/architecture/data changes:

    python tools/architecture/build.py

It describes the whole stack, so it's behind Cloudflare Access as /tls/ is (ui.tls.access): a request through the
tunnel needs an Access token (the CF_Authorization cookie a sign-in at /tls/ leaves serves too); one straight to the
server (this computer, the home network) needs none, as for the rest of the page.
"""
from __future__ import annotations

from .. import paths

PAGE = paths.LAB / "architecture" / "index.html"
SAFE = {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY"}


def handle(handler, path: str) -> bool:
    """Answer a GET for `path` if it's /architecture...: True if answered."""
    if path != "/architecture" and not path.startswith("/architecture/"):
        return False
    from .tls import access

    why = access.refused(handler.headers)
    if why:
        handler._send(403, (why + " The architecture map is behind Cloudflare Access as /tls/ is: sign in at /tls/ "
                            "first, or add /architecture to the Access application.").encode(),
                      "text/plain; charset=utf-8", SAFE)
    elif path == "/architecture":
        handler._send(301, b"", "text/plain", {"Location": "/architecture/"})
    elif path != "/architecture/":
        handler._send(404, b"not found", "text/plain")
    else:
        try:
            body = PAGE.read_bytes()
        except OSError:
            handler._send(404, b"The architecture map isn't built yet: python tools/architecture/build.py",
                          "text/plain; charset=utf-8")
        else:
            handler._send(200, body, "text/html; charset=utf-8", SAFE)
    return True
