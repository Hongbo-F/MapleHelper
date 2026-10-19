"""Reading the game's .wz packages (Wizet "PKG1"), read-only.

The client splits each category into packages: `X.wz` (an index), `X_000.wz` (the data) and `_Canvas/` (its
pictures). A package holds folders and images; an image is a tree of properties: numbers, text, pictures
(canvases), vectors, links to other properties, and sub-trees. Classic World XOR-masks the text but doesn't
AES-encrypt it, so no key is needed.

Offsets in a package's directory are scrambled with a hash of the client version, which isn't stored; only an
8-bit check of it is. The version is found by trying them until every offset lands on an image.
"""

import mmap
import struct
from dataclasses import dataclass, field

OFFSET_KEY = 0x581C3F6D
MAX_VERSION = 10_000


@dataclass
class Canvas:
    """A picture: its size, pixel format and properties (origin, delay...). The pixels are left unread."""
    width: int
    height: int
    fmt: int
    props: dict = field(default_factory=dict)

    def __repr__(self):
        extra = f" {self.props}" if self.props else ""
        return f"<Canvas {self.width}x{self.height} fmt={self.fmt}{extra}>"


@dataclass(frozen=True)
class Vector:
    x: int
    y: int


@dataclass(frozen=True)
class Uol:
    """A link to another property, as a path relative to the link's parent ("../0/body")."""
    path: str


def jsonable(value):
    """A property tree as plain JSON values: pictures become {"_canvas": [w, h, format], ...their properties},
    vectors {"_vec": [x, y]} and links {"_uol": path}."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, Canvas):
        return {"_canvas": [value.width, value.height, value.fmt], **jsonable(value.props)}
    if isinstance(value, Vector):
        return {"_vec": [value.x, value.y]}
    if isinstance(value, Uol):
        return {"_uol": value.path}
    return value


class Reader:
    """Image parsing over a byte buffer: a whole .wz file, or one image decrypted from a .ms pack."""

    def __init__(self, data):
        self.data = data

    def tree(self, base: int = 0, size: int | None = None):
        """The property tree of the image that starts at `base`."""
        typename, p = self._typename(base, base)
        end = base + size if size is not None else len(self.data)
        return self._extended(typename, p, base, end)

    # --- primitives ---

    def _cint(self, p: int) -> tuple[int, int]:
        # A "compressed" int: one signed byte, or -128 and then four.
        v = struct.unpack_from("<b", self.data, p)[0]
        if v == -128:
            return struct.unpack_from("<i", self.data, p + 1)[0], p + 5
        return v, p + 1

    def _clong(self, p: int) -> tuple[int, int]:
        v = struct.unpack_from("<b", self.data, p)[0]
        if v == -128:
            return struct.unpack_from("<q", self.data, p + 1)[0], p + 9
        return v, p + 1

    def _string(self, p: int) -> tuple[str, int]:
        # Length first: negative for one byte per character, positive for UTF-16. Each character is XORed
        # with a mask that counts up from 0xAA (0xAAAA for UTF-16).
        n = struct.unpack_from("<b", self.data, p)[0]
        p += 1
        if n == 0:
            return "", p
        if n > 0:
            if n == 127:
                n = struct.unpack_from("<i", self.data, p)[0]
                p += 4
            units = struct.unpack_from(f"<{n}H", self.data, p)
            return "".join(chr(u ^ ((0xAAAA + i) & 0xFFFF)) for i, u in enumerate(units)), p + 2 * n
        if n == -128:
            n = struct.unpack_from("<i", self.data, p)[0]
            p += 4
        else:
            n = -n
        raw = self.data[p:p + n]
        return "".join(chr(b ^ ((0xAA + i) & 0xFF)) for i, b in enumerate(raw)), p + n

    def _block(self, p: int, base: int) -> tuple[str, int]:
        # A property's name or text: inline (0x00, 0x73), or at an offset from the image start (0x01, 0x1B).
        tag = self.data[p]
        if tag in (0x00, 0x73):
            return self._string(p + 1)
        if tag in (0x01, 0x1B):
            ref = struct.unpack_from("<i", self.data, p + 1)[0]
            return self._string(base + ref)[0], p + 5
        raise ValueError(f"unknown string tag {tag:#x} at {p}")

    def _typename(self, p: int, base: int) -> tuple[str, int]:
        tag = self.data[p]
        if tag == 0x73:
            return self._string(p + 1)
        if tag == 0x1B:
            ref = struct.unpack_from("<i", self.data, p + 1)[0]
            return self._string(base + ref)[0], p + 5
        raise ValueError(f"unknown type tag {tag:#x} at {p}")

    # --- properties ---

    def _properties(self, p: int, base: int) -> tuple[dict, int]:
        out = {}
        count, p = self._cint(p)
        for _ in range(count):
            name, p = self._block(p, base)
            kind = self.data[p]
            p += 1
            if kind == 0:
                value = None
            elif kind in (2, 11):
                value = struct.unpack_from("<h", self.data, p)[0]
                p += 2
            elif kind in (3, 19):
                value, p = self._cint(p)
            elif kind == 20:
                value, p = self._clong(p)
            elif kind == 4:
                # 0x80 then the float; anything else means 0.
                value = 0.0
                if self.data[p] == 0x80:
                    value = struct.unpack_from("<f", self.data, p + 1)[0]
                    p += 4
                p += 1
            elif kind == 5:
                value = struct.unpack_from("<d", self.data, p)[0]
                p += 8
            elif kind == 8:
                value, p = self._block(p, base)
            elif kind == 9:
                # A sub-tree, picture, vector, link or sound, prefixed by its length so unknown kinds can be skipped.
                length = struct.unpack_from("<I", self.data, p)[0]
                end = p + 4 + length
                typename, q = self._typename(p + 4, base)
                value = self._extended(typename, q, base, end)
                p = end
            else:
                raise ValueError(f"unknown property type {kind} at {p}")
            out[name] = value
        return out, p

    def _extended(self, typename: str, p: int, base: int, end: int):
        if typename == "Property":
            return self._properties(p + 2, base)[0]
        if typename == "Canvas":
            has_props = self.data[p + 1]
            p += 2
            props = {}
            if has_props:
                props, p = self._properties(p + 2, base)
            w, p = self._cint(p)
            h, p = self._cint(p)
            fmt, p = self._cint(p)
            return Canvas(w, h, fmt, props)
        if typename == "Shape2D#Vector2D":
            x, p = self._cint(p)
            y, p = self._cint(p)
            return Vector(x, y)
        if typename == "Shape2D#Convex2D":
            count, p = self._cint(p)
            points = []
            for _ in range(count):
                _, p = self._typename(p, base)
                x, p = self._cint(p)
                y, p = self._cint(p)
                points.append(Vector(x, y))
            return points
        if typename == "UOL":
            return Uol(self._block(p + 1, base)[0])
        return f"<{typename}>"  # sounds and raw data (the embedded font), skipped by their length


class Package(Reader):
    """A .wz file: `images()` lists its images by path ("Folder/Name.img"), `image(path)` parses one."""

    def __init__(self, path):
        self.path = str(path)
        with open(path, "rb") as f:
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if f.seek(0, 2) else b""
        super().__init__(data)
        if self.data[:4] != b"PKG1":
            raise ValueError(f"{self.path}: not a .wz package")
        self.start = struct.unpack_from("<I", self.data, 12)[0]
        self.version_check = struct.unpack_from("<H", self.data, self.start)[0]
        self.version: int | None = None
        self.entries: dict[str, tuple[int, int, int]] = {}  # path -> (kind: 3 folder / 4 image, size, offset)
        listing = self._listing(self.start + 2, "")
        if listing:
            self._resolve(listing)

    def images(self) -> list[str]:
        return [p for p, (kind, _, _) in self.entries.items() if kind == 4]

    def image(self, path: str):
        _, size, offset = self.entries[path]
        return self.tree(offset, size)

    # --- directory ---

    def _listing(self, p: int, prefix: str) -> list[tuple[int, str, int, int]]:
        # (kind, path, size, where its scrambled offset is) for each entry of the folder at `p`.
        out = []
        count, p = self._cint(p)
        for _ in range(count):
            kind = self.data[p]
            p += 1
            if kind == 1:
                p += 10
                continue
            if kind == 2:  # the name is stored elsewhere, at an offset from the data start
                ref = struct.unpack_from("<i", self.data, p)[0]
                p += 4
                kind = self.data[self.start + ref]
                name, _ = self._string(self.start + ref + 1)
            else:
                name, p = self._string(p)
            size, p = self._cint(p)
            _checksum, p = self._cint(p)
            out.append((kind, prefix + name, size, p))
            p += 4
        return out

    @staticmethod
    def version_hash(version: int) -> int:
        h = 0
        for c in str(version):
            h = (32 * h + ord(c) + 1) & 0xFFFFFFFF
        return h

    @staticmethod
    def version_check_of(h: int) -> int:
        return 0xFF ^ (h >> 24) ^ ((h >> 16) & 0xFF) ^ ((h >> 8) & 0xFF) ^ (h & 0xFF)

    def scramble(self, p: int, h: int) -> int:
        """The value XORed with the offset stored at `p` (then shifted by twice the data start)."""
        off = ((p - self.start) ^ 0xFFFFFFFF) & 0xFFFFFFFF
        off = (off * h) & 0xFFFFFFFF
        off = (off - OFFSET_KEY) & 0xFFFFFFFF
        r = off & 0x1F
        return ((off << r) | (off >> (32 - r))) & 0xFFFFFFFF if r else off

    def _offset(self, p: int, h: int) -> int:
        stored = struct.unpack_from("<I", self.data, p)[0]
        return ((self.scramble(p, h) ^ stored) + 2 * self.start) & 0xFFFFFFFF

    def _resolve(self, listing):
        for version in range(1, MAX_VERSION):
            h = self.version_hash(version)
            if self.version_check_of(h) != self.version_check:
                continue
            try:
                entries = self._walk(listing, h)
            except (struct.error, IndexError, ValueError):
                continue
            self.version, self.entries = version, entries
            return
        raise ValueError(f"{self.path}: no client version fits its offsets")

    def _walk(self, listing, h) -> dict:
        entries = {}
        for kind, path, size, at in listing:
            offset = self._offset(at, h)
            if not (self.start <= offset and offset + max(size, 0) <= len(self.data)):
                raise ValueError("offset outside the file")
            if kind == 4:
                if size and self.data[offset] != 0x73:
                    raise ValueError("offset isn't an image")
                entries[path] = (4, size, offset)
            elif kind == 3:
                entries[path + "/"] = (3, size, offset)
                entries.update(self._walk(self._listing(offset, path + "/"), h))
        return entries
