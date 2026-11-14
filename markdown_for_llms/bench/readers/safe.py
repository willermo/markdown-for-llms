"""Bounded reading of untrusted archives and XML.

Limits on members, decompressed bytes and compression ratio; member names confined to the
archive; XML with an internal DTD subset (entity declarations) refused before parsing, so no
entity expansion and no external entity can happen; nothing is fetched from the network.
"""
import io
import posixpath
import tarfile
import xml.etree.ElementTree as ET
import zipfile

from ..codes import BenchError, Code

MAX_MEMBERS = 10000
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_RATIO = 200


def _safe_name(name):
    if not name or name.startswith("/") or "\\" in name or "\0" in name:
        return False
    norm = posixpath.normpath(name)
    return not (norm.startswith("../") or norm == ".." or "/../" in f"/{norm}/")


class SafeZip:
    def __init__(self, path):
        try:
            self.zip = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise BenchError(Code.TWIN_UNREADABLE, type(exc).__name__) from None
        infos = self.zip.infolist()
        if len(infos) > MAX_MEMBERS:
            raise BenchError(Code.ARCHIVE_UNSAFE, "too many members")
        total = 0
        self.infos = {}
        for info in infos:
            if not _safe_name(info.filename):
                raise BenchError(Code.ARCHIVE_UNSAFE, "member name")
            if info.filename in self.infos:
                raise BenchError(Code.ARCHIVE_UNSAFE, "duplicate member")
            total += info.file_size
            if info.file_size > MAX_MEMBER_BYTES or total > MAX_TOTAL_BYTES:
                raise BenchError(Code.ARCHIVE_UNSAFE, "declared size")
            if info.compress_size and info.file_size / info.compress_size > MAX_RATIO and info.file_size > 1 << 20:
                raise BenchError(Code.ARCHIVE_UNSAFE, "compression ratio")
            self.infos[info.filename] = info

    def names(self):
        return list(self.infos)

    def has(self, name):
        return name in self.infos

    def read(self, name):
        info = self.infos.get(name)
        if info is None:
            raise BenchError(Code.TWIN_UNREADABLE, "missing member")
        with self.zip.open(info) as fh:
            data = fh.read(MAX_MEMBER_BYTES + 1)
        if len(data) > MAX_MEMBER_BYTES or len(data) != info.file_size:
            raise BenchError(Code.ARCHIVE_UNSAFE, "actual size")
        return data


def read_tar(path, wanted):
    """Members of a (gzipped) tar by name; only regular files inside the archive."""
    out = {}
    try:
        with tarfile.open(path, mode="r:*") as tar:
            members = tar.getmembers()
            if len(members) > MAX_MEMBERS:
                raise BenchError(Code.ARCHIVE_UNSAFE, "too many members")
            total = 0
            for m in members:
                if not _safe_name(m.name):
                    raise BenchError(Code.ARCHIVE_UNSAFE, "member name")
                if m.name in wanted:
                    if not m.isfile():
                        raise BenchError(Code.ARCHIVE_UNSAFE, "not a regular file")
                    total += m.size
                    if m.size > MAX_MEMBER_BYTES or total > MAX_TOTAL_BYTES:
                        raise BenchError(Code.ARCHIVE_UNSAFE, "size")
                    out[m.name] = tar.extractfile(m).read(MAX_MEMBER_BYTES + 1)
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise BenchError(Code.TWIN_UNREADABLE, type(exc).__name__) from None
    missing = set(wanted) - set(out)
    if missing:
        raise BenchError(Code.TWIN_UNREADABLE, "missing member")
    return out


def parse_xml(data: bytes):
    head = data[:65536]
    if b"<!ENTITY" in data or (b"<!DOCTYPE" in head and b"[" in head[head.find(b"<!DOCTYPE"):head.find(b">", head.find(b"<!DOCTYPE")) + 1]):
        raise BenchError(Code.XML_UNSAFE, "internal DTD subset")
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise BenchError(Code.TWIN_UNREADABLE, "XML: " + str(exc)[:80]) from None


def local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def text_of(el, skip=()):
    """Text of an element and its descendants, skipping subtrees whose local tag is in `skip`."""
    parts = []

    def rec(x):
        if local(x.tag) in skip:
            return
        parts.append(x.text or "")
        for c in x:
            rec(c)
            parts.append(c.tail or "")
    rec(el)
    return "".join(parts)


def bytes_io(data):
    return io.BytesIO(data)
