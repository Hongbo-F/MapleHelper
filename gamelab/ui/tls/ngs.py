#!/usr/bin/env python3
# A copy of D:/log decoding .tools/decode_ngs_request.py as of its commit d52ddb8 (2026-11-13), for the /tls/
# page (ui.tls). That repo's copy is where it's worked on; a fix to one belongs in the other.
r"""COMPLETE offline decoder for the NGS "AE: G" HTTPS request bodies.

The keystream function has been fully RECOVERED from grap-updater.aes (the
32-bit updater builds most of these requests; grap-communicator64 uses the
same cipher). It is a tiny self-synchronizing 16-bit-word stream cipher --
no table, no key, no server round-trip: a captured body decodes byte-exactly
from the ciphertext alone.

Target: grap telemetry / error-report requests to
  x-update / x-phaethon / x-csauth / x-init / x-config .ngs.nexon.com
whose HTTP request body carries the header "AE: G".  (The same cipher also
covers a few bodies sent WITHOUT an AE header -- x-phaethon /v1, x-csauth
/v1/close -- so decode is driven by body shape, not the header.)

Wire format (what the base64 body decodes to):
    [1 byte n][n+2 random IV bytes][ciphertext]
    len(ciphertext) == len(plaintext)        (length-preserving)

THE CIPHER (recovered, verified byte-exact on all captures incl. holdout).
Work on little-endian 16-bit WORDS of the plaintext/ciphertext buffer. Let
k be a 16-bit running keystream word:

    k_0   = seed                         (see below)
    ct_j  = pt_j XOR k_j                  (word j)
    k_j+1 = (k_j + ct_j)  mod 2^16        (ciphertext feedback -> self-sync)

i.e. the keystream word is just the seed plus the running sum of all prior
ciphertext words. A trailing odd byte is XORed with the low byte of the
current k. In the binary this is MBA-obfuscated inside a control-flow-
flattened function: the per-word combiner (c|~b)+(~c|b)-2*(~(c|b))-2*(c&b)
is exactly XOR, and the state update reduces to the +ct add above
(grap-updater.aes RVA ~0x10e43c1, loop writing a DWORD = two words per pass).

THE SEED.  k_0 is simply the keystream for the first plaintext word, so it is
pinned by the KNOWN PLAINTEXT, not by the IV.  Every body is a JSON object,
so its first two bytes are '{' and '"' (0x7b, 0x22 -> little-endian word
0x227b), which fixes both bytes of k_0 straight from the ciphertext:

    k_0 = ct_word0 XOR 0x227b
        = (ct[0] ^ ord('{')) | ((ct[1] ^ ord('"')) << 8)

The IV is NOT needed for decoding (the known '{"' prefix plus ciphertext
feedback carries the whole message), but the client's seed->IV derivation has
now also been fully recovered from the code -- see the "IV CONSTRUCTION"
section and mask16()/seed_from_iv() below.  In short: k_0 = MSVC rand()&0x7fff,
and the IV's last two bytes carry it masked by a length tag + endpoint id:
    iv_last2 = k_0 XOR mask16(ct_len) XOR (ENDPOINT_TYPE << 8)
    mask16(len) = len XOR (100*((len//100) % 10) + 70)
So the seed can be recovered from the IV too (seed_from_iv(), the server's
method) -- an alternative to the prefix seed when the endpoint TYPE is known.
An earlier build keyed the seed's high byte off iv[-1]; that was a coincidence
(iv[-1] == true high byte in only 9 of 186 live bodies) -- the real rule is the
mask above.  If a body ever does not start with the expected prefix (e.g. a
JSON array), decode falls back to a 256-way search of the high byte, keeping
the low byte pinned by '{'.

Usage:
    python decode_ngs_request.py -f body.b64            # base64 body
    python decode_ngs_request.py "AAAB...=="
    python decode_ngs_request.py --raw -f body.bin      # raw [n][IV][ct]
    cat body.b64 | python decode_ngs_request.py
    python decode_ngs_request.py --selftest             # verify on the corpus
"""
import argparse
import base64
import glob
import json
import os
import sys

JSON_PREFIX = b'{"'         # every AE:G body is a JSON object; its first two
                            # bytes pin the whole 16-bit seed word k_0.


# --- Blob / request parsing --------------------------------------------------

def parse_blob(blob):
    """Split a raw request blob into (n, iv, ct).  blob = [n][n+2 IV][ciphertext]."""
    if len(blob) < 1:
        raise ValueError("empty request blob")
    n = blob[0]
    iv_len = n + 2
    if len(blob) < 1 + iv_len:
        raise ValueError(
            "blob too short: declares n=%d (IV %d bytes) but only %d bytes after the length byte"
            % (n, iv_len, len(blob) - 1))
    iv = blob[1:1 + iv_len]
    ct = blob[1 + iv_len:]
    return n, iv, ct


# --- The recovered cipher ----------------------------------------------------

def keystream_decrypt(ct, seed):
    """Decrypt ciphertext bytes with the recovered word cipher given the 16-bit seed.
    ct_j = pt_j ^ k_j ; k_{j+1} = (k_j + ct_j) mod 2^16 ; trailing byte uses low(k)."""
    ct = bytes(ct)
    out = bytearray(len(ct))
    k = seed & 0xffff
    nb = len(ct)
    j = 0
    while 2 * j + 1 < nb:
        cw = ct[2 * j] | (ct[2 * j + 1] << 8)
        pw = cw ^ k
        out[2 * j] = pw & 0xff
        out[2 * j + 1] = (pw >> 8) & 0xff
        k = (k + cw) & 0xffff
        j += 1
    if nb % 2:                                  # trailing odd byte
        out[nb - 1] = ct[nb - 1] ^ (k & 0xff)
    return bytes(out)


# --- IV CONSTRUCTION: the client's seed->IV masking (recovered 2026-11-13) ----
# The decoder does NOT need any of this (k_0 comes from the known '{"' prefix),
# but this is the fully-recovered client side -- how grap derives the per-message
# seed and folds it into the random IV so the server can recover it.
#
#   k_0 (cipher seed) = MSVC rand() & 0x7fff
#       LCG state @ grap-updater.aes 0x284b1d4:  s = s*0x343fd + 0x269ec3 ;
#       k_0 = (s >> 16) & 0x7fff.   The IV's first n bytes are rand() too.
#   The IV's LAST two bytes carry the seed, masked by a length tag + endpoint id:
#       iv_last2 = k_0 XOR mask16(ct_len) XOR (ENDPOINT_TYPE << 8)
#   written at grap-updater.aes 0x10e4a77 (computed by 0x10e4dcd..0x10e4a77).
#
#   mask16(len) = len XOR (100*((len//100) % 10) + 70)        # 70 = 0x46
#   The "100*hundreds_digit + 70" carrier is the binary's len%1000 / len%100
#   decimal-digit arithmetic (reciprocals 0x10624dd3 for /1000, 0x147b for /100);
#   its two MBA blocks reduce to plain XOR.  Verified: the low byte equals the
#   capture-derived g(len) for 36/36 distinct lengths, and the FULL 16-bit
#   relation holds for 37/37 corpus bodies (all TYPE 0).

# Per-endpoint TYPE constant, XORed into the HIGH byte of iv_last2.
ENDPOINT_TYPE = {
    "x-csauth/v2":       0x1f,
    "x-update/v2":       0x01,
    "x-csauth/v1/issue": 0x01,
    "x-config/v1":       0x00,
    "x-init/v1":         0x00,
    "x-phaethon/v1":     0x00,
    "x-csauth/v1/close": 0x00,
}


def mask16(ct_len):
    """16-bit length mask folded into the IV's last two bytes (seed carrier)."""
    return (ct_len ^ (100 * ((ct_len // 100) % 10) + 70)) & 0xffff


def g_len(ct_len):
    """Low byte of mask16() -- the server's per-length 'g(len)' tag."""
    return mask16(ct_len) & 0xff


def reconstruct_iv_last2(seed, ct_len, endpoint_type=0):
    """Compute the IV's last 2 bytes the way the CLIENT does (seed-carrying).
    Returns the 16-bit little-endian word (low byte -> iv[-2], high -> iv[-1])."""
    return (seed ^ mask16(ct_len) ^ ((endpoint_type & 0xff) << 8)) & 0xffff


def seed_from_iv(iv, ct_len, endpoint_type=0):
    """Recover k_0 from the IV the way the SERVER does (inverse of the masking).
    Alternative to the known-prefix seed; requires the endpoint TYPE."""
    iv = bytes(iv)
    iv_last2 = iv[-2] | (iv[-1] << 8)
    return (iv_last2 ^ mask16(ct_len) ^ ((endpoint_type & 0xff) << 8)) & 0xffff


def _looks_json(pt):
    """True if pt is a parseable UTF-8 JSON value (the body is always one)."""
    try:
        json.loads(pt.decode("utf-8"))
        return True
    except Exception:  # noqa: BLE001
        return False


def _printable_score(pt):
    """Fraction of bytes that are typical JSON text (fallback ranking)."""
    if not pt:
        return 0.0
    good = sum(1 for b in pt if 0x20 <= b < 0x7f or b in (0x09, 0x0a, 0x0d))
    return good / len(pt)


def _search_high(ct, lo):
    """Pin the seed low byte (from '{') and search all 256 high bytes.
    Return (seed, plaintext) for the best candidate -- a valid-JSON hit if one
    exists, else the most text-like decode."""
    best = None
    for hi in range(256):
        seed = (hi << 8) | lo
        pt = keystream_decrypt(ct, seed)
        if _looks_json(pt):
            return seed, pt                     # unique in practice
        sc = _printable_score(pt)
        if best is None or sc > best[0]:
            best = (sc, seed, pt)
    return best[1], best[2]


class DecodeResult:
    """Byte-exact decode result.

    plaintext : bytes (fully recovered)
    total     : length
    seed      : the 16-bit k_0 used
    method    : "prefix" (k_0 from the known '{"' word) or "search" (high-byte scan)
    is_json   : True if the plaintext parses as JSON (the expected body shape)
    """

    def __init__(self, plaintext, seed, iv, method="prefix"):
        self.plaintext = bytes(plaintext)
        self.seed = seed & 0xffff
        self.total = len(plaintext)
        self.iv = bytes(iv)
        self.method = method
        self.is_json = _looks_json(self.plaintext)
        # back-compat alias (older callers checked .iv_check); now = JSON validity
        self.iv_check = self.is_json
        # back-compat fields (always fully known now)
        self.known = [True] * self.total
        self.n_known = self.total
        self.n_unknown = 0
        self.coverage = 100.0
        self.unknown_indices = []

    def is_complete(self):
        return True

    def report(self):
        chk = "valid JSON" if self.is_json else "NOT valid JSON (unexpected body?)"
        return ("byte-exact: %d/%d bytes, seed k_0=0x%04x (%s), %s"
                % (self.n_known, self.total, self.seed, self.method, chk))


def decrypt(iv, ct, prefix=JSON_PREFIX):
    """Decrypt (iv, ciphertext) -> DecodeResult.  k_0 comes from the known JSON
    prefix ('{"' -> both seed bytes); `iv` is accepted for API compatibility but
    is not used for decoding.  Falls back to a high-byte search if the prefix
    guess does not yield valid JSON."""
    iv = bytes(iv)
    ct = bytes(ct)
    if not ct:
        return DecodeResult(b"", 0, iv)
    prefix = bytes(prefix) or JSON_PREFIX
    lo = ct[0] ^ prefix[0]
    if len(prefix) >= 2 and len(ct) >= 2:
        seed = ((ct[1] ^ prefix[1]) << 8) | lo
        pt = keystream_decrypt(ct, seed)
        if _looks_json(pt):
            return DecodeResult(pt, seed, iv, method="prefix")
        # prefix[1] wrong (e.g. array / odd body) -> search the high byte
    seed, pt = _search_high(ct, lo)
    return DecodeResult(pt, seed, iv, method="search")


def decode_blob(blob, prefix=JSON_PREFIX):
    """Decode a raw [n][IV][ct] request blob.  Returns a DecodeResult."""
    _n, iv, ct = parse_blob(blob)
    return decrypt(iv, ct, prefix=prefix)


def decode_body(b64_body, prefix=JSON_PREFIX):
    """Decode a base64 request body (str or bytes).  Returns a DecodeResult."""
    if isinstance(b64_body, (bytes, bytearray)):
        b64_body = bytes(b64_body)
    b64_body = b64_body.strip()
    blob = base64.b64decode(b64_body)
    return decode_blob(blob, prefix=prefix)


# --- Self-test ---------------------------------------------------------------

def _run_dir(label, directory, prefix_glob):
    bins = sorted(glob.glob(os.path.join(directory, prefix_glob + "*.bin")))
    if not bins:
        print("  (no %s pairs at %s)" % (label, directory))
        return (0, 0)
    nmsg = exact = 0
    for b in bins:
        j = b[:-4] + ".json"
        if not os.path.exists(j):
            continue
        nmsg += 1
        blob = open(b, "rb").read()
        want = open(j, "rb").read()
        try:
            res = decode_blob(blob)
        except Exception as e:  # noqa: BLE001
            print("  %-10s ERROR %s" % (os.path.basename(b), e))
            continue
        ok = (res.plaintext == want)
        exact += 1 if ok else 0
        if not ok:
            d = [i for i in range(min(len(want), res.total)) if res.plaintext[i] != want[i]]
            print("  %-10s MISMATCH first bad idx %s" % (os.path.basename(b), d[:3]))
    print("  -> %s: %d/%d messages byte-exact" % (label, exact, nmsg))
    return (nmsg, exact)


def _mask_selftest(cw):
    """Verify the recovered IV masking.  Every corpus body is TYPE 0, so the
    seed recovered FROM the IV (server method) must equal the known-prefix seed,
    AND the full 16-bit iv_last2 must round-trip through reconstruct_iv_last2()."""
    bins = sorted(glob.glob(os.path.join(cw, "pairs", "p*.bin")) +
                  glob.glob(os.path.join(cw, "holdout", "g*.bin")))
    n = ok = 0
    for b in bins:
        try:
            _n, iv, ct = parse_blob(open(b, "rb").read())
        except Exception:  # noqa: BLE001
            continue
        if len(ct) < 2 or len(iv) < 2:
            continue
        n += 1
        k0_prefix = (ct[0] ^ 0x7b) | ((ct[1] ^ 0x22) << 8)
        k0_iv = seed_from_iv(iv, len(ct), endpoint_type=0)
        iv_last2 = iv[-2] | (iv[-1] << 8)
        if k0_iv == k0_prefix and reconstruct_iv_last2(k0_prefix, len(ct), 0) == iv_last2:
            ok += 1
        else:
            print("  %-10s IV-MASK MISMATCH len=%d prefix=0x%04x fromIV=0x%04x"
                  % (os.path.basename(b), len(ct), k0_prefix, k0_iv))
    print("  -> IV masking (seed_from_iv == prefix seed, full 16-bit, TYPE 0): %d/%d" % (ok, n))
    return n > 0 and ok == n


def _selftest():
    here = os.path.dirname(os.path.abspath(__file__))
    cw = os.path.join(here, "cipher_work")
    print("Recovered cipher (word stream: ct=pt^k, k+=ct; seed k_0 from the '{\"' prefix).\n")
    n1, e1 = _run_dir("in-corpus pairs", os.path.join(cw, "pairs"), "p")
    n2, e2 = _run_dir("holdout", os.path.join(cw, "holdout"), "g")
    total, exact = n1 + n2, e1 + e2
    print("\nDecode: %d/%d messages byte-exact." % (exact, total))
    print("\nIV masking (mask16 / seed_from_iv, recovered from code):")
    mask_ok = _mask_selftest(cw)
    print("\nTOTAL: decode %d/%d byte-exact; IV-mask check %s."
          % (exact, total, "PASS" if mask_ok else "FAIL"))
    return total > 0 and exact == total and mask_ok


# --- CLI ---------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="COMPLETE decoder for NGS 'AE: G' request bodies (grap). "
                    "Recovered self-synchronizing 16-bit word stream cipher; "
                    "decodes byte-exactly from the ciphertext alone.")
    ap.add_argument("body", nargs="?",
                    help="base64 request body. If omitted and no --selftest, read stdin.")
    ap.add_argument("-f", "--file", help="read the body from this file")
    ap.add_argument("--raw", action="store_true",
                    help="input is the raw [n][IV][ct] blob (bytes), not base64. Use with -f or stdin.")
    ap.add_argument("--prefix", default='{"',
                    help="known plaintext prefix used to seed k_0 (default '{\"', the JSON object start)")
    ap.add_argument("--selftest", action="store_true",
                    help="verify byte-exact decode on the in-corpus pairs and the holdout")
    ap.add_argument("-o", "--out", help="write decoded plaintext to this file instead of stdout")
    args = ap.parse_args(argv)

    if args.selftest:
        return 0 if _selftest() else 1

    prefix = args.prefix.encode() if isinstance(args.prefix, str) else bytes(args.prefix)
    if not prefix:
        prefix = JSON_PREFIX

    if args.raw:
        if args.file:
            data = open(args.file, "rb").read()
        elif args.body is not None:
            data = bytes.fromhex(args.body.strip())
        else:
            data = sys.stdin.buffer.read()
        res = decode_blob(data, prefix=prefix)
    else:
        if args.file:
            b64 = open(args.file, "rb").read()
        elif args.body is not None:
            b64 = args.body
        else:
            b64 = sys.stdin.buffer.read()
        res = decode_body(b64, prefix=prefix)

    print(res.report(), file=sys.stderr)
    if args.out:
        with open(args.out, "wb") as fh:
            fh.write(res.plaintext)
        print("wrote %d bytes to %s" % (len(res.plaintext), args.out), file=sys.stderr)
    else:
        sys.stdout.buffer.write(res.plaintext)
        if sys.stdout.isatty():
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
