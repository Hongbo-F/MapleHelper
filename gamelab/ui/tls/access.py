"""Who may read /tls/: the decrypted captures hold live session tokens (csauthtoken, lifetoken, the launcher's cookies),
the HWID and the machine's inventory, and the mini-map's server asks for no password (ui.web), while its tunnel puts it
on the internet (maple.eirage.com). So a request that came through Cloudflare must carry a Cloudflare Access token,
checked here (the user, 2026-11-13, of the ways offered: "Cloudflare Access"); one straight to the server (this
computer, the home network) needs none, as for the rest of the page.

Through Cloudflare: Cloudflare's edge adds Cf-Ray / Cf-Connecting-IP / Cdn-Loop to every request it passes on, and a
browser can't take them off; a request without them came straight to the server.

The token: the JWT Access puts in Cf-Access-Jwt-Assertion (and the CF_Authorization cookie), signed with the team's
key (RS256, the keys from https://<team>.cloudflareaccess.com/cdn-cgi/access/certs, fetched when first needed and
again hourly, or for a key not seen), for the application's audience tag, from the team's issuer, not expired. A
header alone proves nothing (a browser can send one), so its signature is what counts. Set up in CONFIG:
    {"team": "<team>" (or "<team>.cloudflareaccess.com"), "aud": "<the Access application's Application Audience tag>"}
Without it every request through Cloudflare is refused: fail closed."""
from __future__ import annotations

import base64
import json
import threading
import time
import urllib.request
from http.cookies import SimpleCookie

from ... import paths

CONFIG = paths.LAB / "cloudflare_access.json"
KEYS_TTL_S = 3600        # the team's keys are fetched again after this
REFETCH_S = 60           # ... or for a key not among them, at most this often
LEEWAY_S = 60            # clocks apart: a token this far before its nbf, or past its exp, still passes
_keys: dict[str, tuple[float, dict]] = {}   # team URL -> (fetched at, {kid: public key})
_keys_lock = threading.Lock()


class Refused(Exception):
    pass


def through_cloudflare(headers) -> bool:
    return bool(headers.get("Cf-Ray") or headers.get("Cf-Connecting-IP") or headers.get("Cf-Visitor")
                or "cloudflare" in (headers.get("Cdn-Loop") or "").lower())


def config(path=None) -> dict | None:
    """{"team": the team's URL, "aud": its audience tag} from CONFIG, or None (not there, not complete)."""
    try:
        c = json.loads((path or CONFIG).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    team, aud = str(c.get("team") or "").strip().rstrip("/"), str(c.get("aud") or "").strip()
    if not team or not aud:
        return None
    if "://" not in team:
        team = "https://" + (team if "." in team else team + ".cloudflareaccess.com")
    return {"team": team, "aud": aud}


def refused(headers, now: float | None = None, cfg: dict | None = None) -> str | None:
    """Why this request may not read /tls/, or None if it may."""
    if not through_cloudflare(headers):
        return None
    cfg = cfg or config()
    if cfg is None:
        return (f"Refused: this came through Cloudflare, and Cloudflare Access isn't set up for it (gamelab/lab/"
                f"{CONFIG.name} with the team and the application's audience tag: gamelab/ui/tls/access.py).")
    token = headers.get("Cf-Access-Jwt-Assertion") or _cookie(headers.get("Cookie"), "CF_Authorization")
    if not token:
        return "Refused: no Cloudflare Access token (is an Access application in front of this address's /tls/?)."
    try:
        verify(token, cfg["team"], cfg["aud"], time.time() if now is None else now)
    except Refused as e:
        return f"Refused: the Cloudflare Access token {e}."
    return None


def _cookie(header: str | None, name: str) -> str | None:
    if not header:
        return None
    try:
        c = SimpleCookie()
        c.load(header)
    except Exception:  # noqa: BLE001 (a cookie header the parser can't take)
        return None
    return c[name].value if name in c else None


def _b64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def verify(token: str, team: str, aud: str, now: float, keys=None) -> dict:
    """The token's claims if Access signed it for `aud`, else Refused. `keys(team, kid)` gives the public key (default:
    the team's, fetched)."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    try:
        h64, p64, s64 = token.split(".")
        header, claims, sig = json.loads(_b64(h64)), json.loads(_b64(p64)), _b64(s64)
    except ValueError:
        raise Refused("is not a JWT") from None
    if header.get("alg") != "RS256":
        raise Refused(f"is signed with {header.get('alg')!r}, not RS256")
    key = (keys or team_key)(team, header.get("kid"))
    if key is None:
        raise Refused("is signed with a key the team doesn't have")
    try:
        key.verify(sig, f"{h64}.{p64}".encode(), padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature:
        raise Refused("has a bad signature") from None
    auds = claims.get("aud")
    if aud not in (auds if isinstance(auds, list) else [auds]):
        raise Refused("is for another Access application")
    if claims.get("iss") != team:
        raise Refused(f"is from {claims.get('iss')!r}, not {team}")
    if not isinstance(claims.get("exp"), (int, float)) or now > claims["exp"] + LEEWAY_S:
        raise Refused("has expired")
    if isinstance(claims.get("nbf"), (int, float)) and now < claims["nbf"] - LEEWAY_S:
        raise Refused("isn't valid yet")
    return claims


def team_key(team: str, kid):
    """The team's public key `kid`, from its certs (kept KEYS_TTL_S; fetched again for a kid not among them, at most
    every REFETCH_S), or None."""
    with _keys_lock:
        at, keys = _keys.get(team, (0.0, {}))
        age = time.time() - at
        if age > KEYS_TTL_S or (kid not in keys and age > REFETCH_S):
            try:
                keys = _fetch(team)
            except Exception:  # noqa: BLE001 (the team unreachable: the keys there are, until the next try)
                keys = keys or {}
            _keys[team] = (time.time(), keys)
        return keys.get(kid)


def _fetch(team: str) -> dict:
    from cryptography.hazmat.primitives.asymmetric import rsa

    with urllib.request.urlopen(f"{team}/cdn-cgi/access/certs", timeout=5) as r:
        jwks = json.loads(r.read())
    out = {}
    for k in jwks.get("keys", []):
        if k.get("kty") == "RSA" and k.get("kid"):
            out[k["kid"]] = rsa.RSAPublicNumbers(int.from_bytes(_b64(k["e"]), "big"),
                                                 int.from_bytes(_b64(k["n"]), "big")).public_key()
    return out
