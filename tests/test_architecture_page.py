"""The architecture map in the web UI (gamelab.ui.architecture): /architecture/ serves the built page, straight to the
server freely and through the tunnel only with a Cloudflare Access token, and nothing else under it."""
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from gamelab.ui import architecture, web
from gamelab.ui.tls import access

CF = {"Cf-Ray": "8a1b2c3d4e5f-NRT", "Cf-Connecting-IP": "203.0.113.9", "Cdn-Loop": "cloudflare"}


@pytest.fixture
def served(monkeypatch, tmp_path):
    page = tmp_path / "architecture" / "index.html"
    page.parent.mkdir()
    page.write_text("<!doctype html><title>Autonomy Player Map</title>", encoding="utf-8")
    monkeypatch.setattr(architecture, "PAGE", page)
    monkeypatch.setattr(access, "CONFIG", tmp_path / "no-access.json")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web._Handler)
    srv.feed = None  # (only /architecture is asked)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", page
    srv.shutdown()
    srv.server_close()


def get(url, headers=None):
    r = urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}))
    return r.status, r.headers, r.read()


def status(url, headers=None):
    try:
        return get(url, headers)[0]
    except urllib.error.HTTPError as e:
        return e.code


def test_the_built_page_is_served_straight_to_the_server(served):
    base, page = served
    code, headers, body = get(base + "/architecture/")
    assert code == 200 and body == page.read_bytes()
    assert headers["Content-Type"].startswith("text/html")
    assert get(base + "/architecture")[2] == page.read_bytes()   # the redirect to /architecture/ followed


def test_through_the_tunnel_without_access_it_is_refused(served):
    base, _ = served
    assert status(base + "/architecture/", CF) == 403
    assert status(base + "/architecture", CF) == 403


def test_nothing_else_under_it_and_a_page_not_built_says_so(served):
    base, page = served
    for path in ("/architecture/../web.py", "/architecture/index.html", "/architecture/data/x.json"):
        assert status(base + path) == 404, path
    page.unlink()
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base + "/architecture/")
    assert e.value.code == 404 and b"build.py" in e.value.read()
