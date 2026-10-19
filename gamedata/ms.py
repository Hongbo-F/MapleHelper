"""Reading the game's .ms packs (format version 4), read-only.

Newer clients keep some categories in `Data/Packs/*.ms` instead of .wz; Classic World does so for Mob and Skill.
A pack is encrypted with ChaCha20, with keys derived from its file name and a salt stored in it: first an
8-byte header, then a table of images, then the images, of which only the first 1 KB is encrypted. Inside, an
image is an ordinary .wz image.

The format follows WzComparerR2's Ms_FileV2 and Ms_ImageV2 (https://github.com/Kagamia/WzComparerR2, MIT).
"""

import mmap
import os
import struct

import numpy as np

from .wz import Reader

# XORed into every key.
KEY_OBSCURE = bytes([
    0x7B, 0x2F, 0x35, 0x48, 0x43, 0x95, 0x02, 0xB9, 0xAE, 0x91, 0xA6, 0xE1, 0xD8, 0xD6, 0x24, 0xB4,
    0x33, 0x10, 0x1D, 0x3D, 0xC1, 0xBB, 0xC6, 0xF4, 0xA5, 0xFE, 0xB3, 0x69, 0x6B, 0x56, 0xE4, 0x75,
])
_SIGMA = np.frombuffer(b"expand 32-byte k", "<u4")


def chacha20(key: bytes, nonce: bytes, counter: int, blocks: int) -> bytes:
    """`blocks` 64-byte blocks of ChaCha20 keystream (RFC 7539: 32-byte key, 12-byte nonce), from block `counter`."""
    state = np.zeros((16, blocks), np.uint32)
    state[0:4] = _SIGMA[:, None]
    state[4:12] = np.frombuffer(key, "<u4")[:, None]
    n = np.arange(blocks, dtype=np.uint64) + counter
    state[12] = (n & 0xFFFFFFFF).astype(np.uint32)
    words = np.frombuffer(nonce, "<u4")
    state[13] = (words[0] + (n >> 32)).astype(np.uint32)  # the counter carries into the nonce, as the game's does
    state[14], state[15] = words[1], words[2]
    x = state.copy()

    def rotl(v, s):
        return (v << np.uint32(s)) | (v >> np.uint32(32 - s))

    def quarter(a, b, c, d):
        x[a] += x[b]
        x[d] = rotl(x[d] ^ x[a], 16)
        x[c] += x[d]
        x[b] = rotl(x[b] ^ x[c], 12)
        x[a] += x[b]
        x[d] = rotl(x[d] ^ x[a], 8)
        x[c] += x[d]
        x[b] = rotl(x[b] ^ x[c], 7)

    with np.errstate(over="ignore"):
        for _ in range(10):
            quarter(0, 4, 8, 12), quarter(1, 5, 9, 13), quarter(2, 6, 10, 14), quarter(3, 7, 11, 15)
            quarter(0, 5, 10, 15), quarter(1, 6, 11, 12), quarter(2, 7, 8, 13), quarter(3, 4, 9, 14)
        x += state
    return x.T.astype("<u4").tobytes()


def xor(data: bytes, stream: bytes) -> bytes:
    return (np.frombuffer(data, np.uint8) ^ np.frombuffer(stream[:len(data)], np.uint8)).tobytes()


class TableReader:
    """Decrypts the image table as the game reads it: in 64-byte blocks, where the block counter goes back to 0
    whenever a read ends exactly at the end of a block. The format depends on that quirk."""

    def __init__(self, data, pos: int, key: bytes):
        self.data, self.pos, self.key = data, pos, key
        self.counter, self.block, self.used = 0, b"", 64

    def read(self, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            if self.used >= 64:
                self.block = xor(self.data[self.pos:self.pos + 64], chacha20(self.key, bytes(12), self.counter, 1))
                self.pos += 64
                self.counter = (self.counter + 1) & 0xFFFFFFFF
                self.used = 0
            take = min(n, 64 - self.used)
            out += self.block[self.used:self.used + take]
            self.used += take
            n -= take
        if self.used >= 64:
            self.counter = 0
        return bytes(out)

    def int32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def string(self) -> str:
        return self.read(2 * self.int32()).decode("utf-16-le")


def _signed_half(b: int) -> int:
    return ((b - 256 if b > 127 else b) >> 1) & 0xFF


def header_key(name_with_salt: str) -> bytes:
    s = name_with_salt
    return bytes(((ord(s[i % len(s)]) + i) & 0xFF) ^ KEY_OBSCURE[i] for i in range(32))


def table_key(name_with_salt: str) -> bytes:
    s = name_with_salt
    return bytes(((i + (i % 3 + 2) * ord(s[len(s) - 1 - i % len(s)])) & 0xFF) ^ KEY_OBSCURE[i] for i in range(32))


def image_cipher(salt: str, name: str, entry_key: bytes) -> tuple[bytes, bytes, int]:
    """(key, nonce, first block counter) for one image, from the pack's salt, the image's path and its key."""
    h = 0x811C9DC5
    for c in salt:
        h = ((h ^ ord(c)) * 0x1000193) & 0xFFFFFFFF
    h2 = h >> 1
    h3 = h2 ^ 0x6C
    digits = [int(c) for c in str(h)]
    nd = len(digits)
    key = bytes(
        ((i + ord(name[i % len(name)]) * (digits[i % nd] % 2 + entry_key[(digits[(i + 2) % nd] + i) % 16]
                                          + (digits[(i + 1) % nd] + i) % 5)) & 0xFF) ^ KEY_OBSCURE[i]
        for i in range(32))
    mixed = bytearray(struct.pack("<III", h, h2, h3))
    a = b = d = 0
    c = 90
    for i in range(12):
        mixed[i] ^= (d + 11 * (i // 11) + (c ^ (i >> 2)) + (a ^ b)) & 0xFF
        d -= 1
        a += 8
        b += 17
        c += 43
    return key, bytes(4) + bytes(mixed[:8]), struct.unpack("<I", bytes(mixed[8:12]))[0]


class MsPack:
    """A .ms file: `images()` lists its images by path ("Mob/0000002.img"), `image(path)` parses one."""

    VERSION = 4

    def __init__(self, path):
        self.path = str(path)
        with open(path, "rb") as f:
            d = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if f.seek(0, 2) else b""
        self.data = d
        name = os.path.basename(self.path).lower()
        # Random filler whose length comes from the file name, then the version and the salt, masked with it.
        filler = [_signed_half(b) for b in d[:sum(map(ord, name)) % 312 + 30]]
        p = len(filler)
        if p + 5 > len(d):
            raise ValueError(f"{self.path}: too short for a .ms pack")
        masked_version = d[p]
        if masked_version ^ filler[0] != self.VERSION:
            raise ValueError(f"{self.path}: not a version {self.VERSION} .ms pack")
        hashed_salt_len = struct.unpack_from("<i", d, p + 1)[0]
        p += 5
        salt_len = (hashed_salt_len & 0xFF) ^ filler[0]
        salt_raw = d[p:p + 2 * salt_len]
        p += 2 * salt_len
        chars = []
        for i in range(salt_len):
            a = filler[i] ^ salt_raw[2 * i]
            chars.append(chr(((a | 0x4B) << 1) - a - 75))
        self.salt = "".join(chars)
        name_salt = name + self.salt

        hash_, count = struct.unpack("<ii", xor(d[p:p + 8], chacha20(header_key(name_salt), bytes(12), 0, 1)))
        check = hashed_salt_len + masked_version + self.VERSION + count + sum(struct.unpack(f"<{salt_len}H", salt_raw))
        if (check - hash_) & 0xFFFFFFFF:
            raise ValueError(f"{self.path}: header check failed (wrong file name?)")

        table = TableReader(d, p + 8 + sum(ord(c) * 3 for c in name) % 212 + 64, table_key(name_salt))
        rows = []
        for _ in range(count):
            image = table.string()
            _checksum, _flags, start, size, _aligned, _unk1, _unk2 = (table.int32() for _ in range(7))
            key = table.read(16)
            table.int32(), table.int32()
            rows.append((image, start, size, key))
        data_start = -(-table.pos // 1024) * 1024  # images start at the next whole KB
        self.entries = {image: (data_start + start * 1024, size, key) for image, start, size, key in rows}

    def images(self) -> list[str]:
        return list(self.entries)

    def image_bytes(self, path: str) -> bytes:
        start, size, entry_key = self.entries[path]
        key, nonce, counter = image_cipher(self.salt, path, entry_key)
        head = min(size, 1024)
        out = xor(self.data[start:start + head], chacha20(key, nonce, counter, -(-head // 64)))
        return out + self.data[start + 1024:start + size] if size > 1024 else out

    def image(self, path: str):
        return Reader(self.image_bytes(path)).tree()
