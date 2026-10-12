"""Decoder PNG lossless finito per RGB/RGBA/gray8, no interlace.

Asset sintetici e render locali. Nessuna promessa di decoder immagine generale.
"""
import struct
import zlib
from safety import Unsafe, png, MAX_PIXELS


def decode(data):
    width, height = png(data)
    pos, payload, header, ended = 8, bytearray(), None, False
    while pos < len(data):
        if pos + 12 > len(data):
            raise Unsafe("PNG troncato")
        length = struct.unpack_from(">I", data, pos)[0]
        tag = data[pos+4:pos+8]
        part = data[pos+8:pos+8+length]
        if len(part) != length or pos + length + 12 > len(data):
            raise Unsafe("PNG chunk troncato")
        crc = struct.unpack_from(">I", data, pos+8+length)[0]
        if zlib.crc32(tag+part) & 0xffffffff != crc:
            raise Unsafe("CRC PNG")
        if tag == b"IHDR":
            if header is not None:
                raise Unsafe("IHDR duplicato")
            header = struct.unpack(">IIBBBBB", part)
        elif tag == b"IDAT":
            payload.extend(part)
        elif tag == b"IEND":
            ended = True
            break
        elif tag[0] < 97 and tag != b"PLTE":
            raise Unsafe("PNG critical unsupported")
        pos += length + 12
    if not ended or header is None or header[2] != 8 or header[3] not in (0, 2, 6) or header[4:] != (0, 0, 0):
        raise Unsafe("PNG formato non supportato")
    channels = {0: 1, 2: 3, 6: 4}[header[3]]
    stride = width * channels
    limit = height * (stride + 1)
    dec = zlib.decompressobj()
    try:
        raw = dec.decompress(payload, limit + 1)
    except zlib.error as exc:
        raise Unsafe('PNG invalid deflate') from exc
    if len(raw) != limit or not dec.eof or dec.unconsumed_tail:
        raise Unsafe("PNG espansione invalida")
    rows, prev = [], bytearray(stride)
    for y in range(height):
        line = raw[y*(stride+1):(y+1)*(stride+1)]
        filt, row = line[0], bytearray(line[1:])
        if filt not in range(5):
            raise Unsafe("PNG filtro")
        for i in range(stride):
            a = row[i-channels] if i >= channels else 0
            b = prev[i]
            c = prev[i-channels] if i >= channels else 0
            if filt == 1:
                add = a
            elif filt == 2:
                add = b
            elif filt == 3:
                add = (a+b)//2
            elif filt == 4:
                v = a+b-c
                pa, pb, pc = abs(v-a), abs(v-b), abs(v-c)
                add = a if pa <= pb and pa <= pc else b if pb <= pc else c
            else:
                add = 0
            row[i] = (row[i] + add) & 255
        rows.append(bytes(row))
        prev = row
    return width, height, channels, rows


def encode(width, height, channels, rows):
    def chunk(tag, part):
        return struct.pack(">I", len(part)) + tag + part + struct.pack(">I", zlib.crc32(tag+part) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, {1:0,3:2,4:6}[channels], 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"".join(b"\0" + r for r in rows), 9)) + chunk(b"IEND", b"")


def crop(data, left, top, right, bottom):
    w, h, channels, rows = decode(data)
    if not 0 <= left < right <= w or not 0 <= top < bottom <= h:
        raise Unsafe("PNG crop range")
    return encode(right-left, bottom-top, channels, [r[left*channels:right*channels] for r in rows[top:bottom]])


def pixel_digest(data):
    from data import digest_bytes
    w, h, channels, rows = decode(data)
    # Normalize only lossless pixel representation, no resizing/color alteration.
    rgba = bytearray()
    for row in rows:
        for i in range(0, len(row), channels):
            p = row[i:i+channels]
            rgba.extend(p + b"\xff" if channels == 3 else p if channels == 4 else p*3+b"\xff")
    return {"width": w, "height": h, "rgba_sha256": digest_bytes(rgba)}
