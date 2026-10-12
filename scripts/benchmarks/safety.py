"""Validazione prima dei tool e audit inerte dell'output; limiti finiti v1."""
import os
from pathlib import Path, PurePosixPath
import re
import stat
import urllib.parse
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

MAX_FILE = 32 * 1024 * 1024
MAX_PIXELS = 12_000_000


class Unsafe(ValueError):
    pass


def relative(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise Unsafe("Path vuoto/invalido")
    decoded = urllib.parse.unquote(value)
    # One decoding pass only. Encoded separators/double encoding are rejected.
    if decoded != value or "\\" in value or ":" in value:
        raise Unsafe("Path encoded/drive/backslash: " + repr(value))
    p = PurePosixPath(value)
    if p.is_absolute() or any(c in ("", ".", "..") for c in value.split("/")):
        raise Unsafe("Path evasivo: " + repr(value))
    return p


def no_symlink_root(root):
    root = Path(root).absolute()
    if ".." in root.parts or any(p.is_symlink() for p in (root, *root.parents)):
        raise Unsafe("Root evasiva")
    if not root.is_dir():
        raise Unsafe("Root directory assente")
    return root


def regular(root, name, limit=MAX_FILE):
    """Controllo lessicale e lstat degli antenati, quindi O_NOFOLLOW.

    Rifiuta nlink>1: per fixture/copie non attesta hardlink esterni dal nome.
    Non resolve/open sul target di un link prodotto dal motore.
    """
    root = no_symlink_root(root)
    components = relative(name).parts
    current = root
    for i, component in enumerate(components):
        current = current / component
        info = current.lstat()
        if i < len(components) - 1:
            if not stat.S_ISDIR(info.st_mode):
                raise Unsafe("Parent non directory regolare: " + name)
        elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            raise Unsafe("File non regolare/link/eccessivo: " + name)
    fd = os.open(current, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino):
            raise Unsafe("File cambiato durante apertura")
        chunks, length = [], 0
        while True:
            chunk = os.read(fd, min(65536, limit + 1 - length))
            if not chunk:
                break
            chunks.append(chunk)
            length += len(chunk)
            if length > limit:
                raise Unsafe("Limite lettura")
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise Unsafe("File mutato durante lettura")
        return b"".join(chunks)
    finally:
        os.close(fd)


def xml(data):
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise Unsafe("DTD/ENTITY vietato")
    try:
        result = ET.fromstring(data)
    except ET.ParseError as exc:
        raise Unsafe("XML malformato") from exc
    if sum(1 for _ in result.iter()) > 20000:
        raise Unsafe("Troppi nodi XML")
    return result


def local_uri(uri, allow_external=True):
    if uri.startswith("#"):
        return None
    # urlsplit rejects malformed brackets, IP literals and NFKC netlocs with
    # ValueError. These are URI refusals, not internal reader failures. Keep
    # this boundary narrow: unrelated projection/programming errors propagate.
    try:
        parsed = urllib.parse.urlsplit(uri)
    except ValueError as exc:
        raise Unsafe("URI malformata: " + repr(uri)) from exc
    if parsed.scheme or parsed.netloc:
        if allow_external and parsed.scheme in ("https", "http", "mailto") and not parsed.username and not parsed.password:
            return None  # inert string, never fetched
        raise Unsafe("URI vietata: " + repr(uri))
    return relative(parsed.path).as_posix()


class PassiveHTML(HTMLParser):
    TAGS = {"html", "head", "meta", "title", "body", "section", "article", "nav", "header", "footer", "main", "h1", "h2", "h3", "h4", "h5", "h6", "p", "br", "a", "img", "span", "div", "table", "caption", "thead", "tbody", "tr", "td", "th", "figure", "figcaption", "aside", "ol", "ul", "li", "sup", "sub", "em", "strong", "code", "pre", "hr", "math", "mrow", "mi", "mn", "mo", "msub", "msup", "mfrac", "msubsup", "mtable", "mtr", "mtd", "munderover", "mtext", "semantics", "annotation", "link"}

    GLOBAL={'id','class','lang','dir','title','role','aria-label','xmlns','xmlns:epub','epub:type'}
    ATTRS={'a':{'href','rel','target','download'},'img':{'src','alt','width','height'},'link':{'href','rel','type'},'meta':{'charset','name','content'},'td':{'colspan','rowspan','headers'},'th':{'colspan','rowspan','headers','scope'},'ol':{'start'},'math':{'display'},'annotation':{'encoding'}}

    def handle_starttag(self, tag, attrs):
        if tag not in self.TAGS:
            raise Unsafe("HTML attivo/sconosciuto: " + tag)
        for key, value in attrs:
            if key not in self.GLOBAL|self.ATTRS.get(tag,set()) and not key.startswith('data-'):
                raise Unsafe('HTML attribute outside passive subset: '+key)
            if key.lower().startswith("on") or key in ("srcdoc", "style"):
                raise Unsafe("Attributo HTML attivo: " + key)
            if tag == "meta" and key == "http-equiv":
                raise Unsafe("Meta HTTP attivo")
            if key in ("src", "href"):
                if not isinstance(value,str):raise Unsafe('Resource URI absent')
                # External hyperlinks permitted as inert data; never resource URI.
                local_uri(value, allow_external=(key == "href" and tag == "a"))


def html(data):
    p = PassiveHTML(convert_charrefs=True)
    p.feed(data.decode("utf-8", errors="strict") if isinstance(data, bytes) else data)
    p.close()


def css(data):
    """Finite static declarations; no escaped identifiers or resource functions."""
    text = data.decode('utf-8', errors='strict')
    if len(data)>16384 or '\\' in text or '/*' in text or '@' in text:
        raise Unsafe('CSS escape/comment/at-rule outside passive subset')
    allowed={'margin','margin-top','padding','padding-left','color','background','border','border-left','border-bottom','border-collapse','border-radius','font','font-family','font-size','font-weight','line-height','text-align','text-anchor','display','grid-template-columns','gap','min-width','max-width','width','height','max-height','overflow','overflow-wrap','white-space','vertical-align','position','top'}
    at=0
    for rule in re.finditer(r'([^{}]+)\{([^{}]*)\}',text):
        if text[at:rule.start()].strip():raise Unsafe('CSS outside finite rule')
        if not re.fullmatch(r'[A-Za-z0-9_.#,:> \t\r\n-]+',rule[1]):raise Unsafe('CSS selector outside passive subset')
        for declaration in rule[2].split(';'):
            if not declaration.strip():continue
            pair=declaration.split(':',1)
            if len(pair)!=2 or pair[0].strip() not in allowed:raise Unsafe('CSS property outside passive subset')
            prop,value=pair[0].strip(),pair[1].strip()
            if prop=='font-family':
                value=re.sub(r'"[A-Za-z0-9 -]+"|\'[A-Za-z0-9 -]+\'','FONT',value)
            if prop=='grid-template-columns':
                value=re.sub(r'minmax\([0-9]+(?:px|em),[0-9]+fr\)','SIZE',value)
            if not re.fullmatch(r'[A-Za-z0-9_#.,% /+−-]+',value):raise Unsafe('CSS value/function outside passive subset')
        at=rule.end()
    if text[at:].strip():raise Unsafe('CSS trailing unknown syntax')


def svg(data):
    root = xml(data)
    if root.tag.rsplit('}',1)[-1]!='svg':raise Unsafe('SVG root required')
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag not in {"svg", "g", "text", "tspan", "rect", "line", "path", "circle", "ellipse", "image", "defs", "clipPath", "title", "desc"}:
            raise Unsafe("SVG nodo vietato: " + tag)
        for key, value in node.attrib.items():
            key = key.rsplit("}", 1)[-1]
            allowed={'id','class','x','y','width','height','viewBox','preserveAspectRatio','fill','stroke','stroke-width','d','cx','cy','r','rx','ry','x1','x2','y1','y2','font-family','font-size','text-anchor','transform','opacity','fill-opacity','stroke-opacity','href'}
            if key not in allowed or '\\' in value or re.search(r'\burl\s*\(',value,re.I):
                raise Unsafe("SVG attributo attivo")
            if key == "href":
                local_uri(value, allow_external=False)
    return root


def zip_members(content):
    try:
        return _zip_members(content)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise Unsafe('Malformed/encrypted ZIP') from exc


def _zip_members(content):
    import io
    if len(content) > 16 * 1024 * 1024:
        raise Unsafe("Archivio troppo grande")
    members, total = {}, 0
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if len(archive.infolist()) > 256:
            raise Unsafe("Troppi membri")
        for member in archive.infolist():
            name = member.filename[:-1] if member.is_dir() else member.filename
            relative(name)
            mode = member.external_attr >> 16
            if name in members or member.flag_bits & 1 or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise Unsafe("Membro duplicato/cifrato/link")
            if member.is_dir():
                members[name] = None
                continue
            if member.file_size > 16 * 1024 * 1024 or member.file_size / max(member.compress_size, 1) > 100:
                raise Unsafe("Limite file/ratio ZIP")
            with archive.open(member) as stream:
                payload = stream.read(16 * 1024 * 1024 + 1)
            total += len(payload)
            if total > 64 * 1024 * 1024 or len(payload) != member.file_size:
                raise Unsafe("Limite espansione ZIP")
            lower=name.casefold()
            if lower.endswith((".xml", ".rels", ".opf", ".xhtml", ".html", ".ncx")):
                tree = xml(payload)
                markup=lower.endswith((".xhtml", ".html")) or tree.tag.rsplit('}',1)[-1]=='html'
                # XML resource containers obey the same pre-tool URI boundary,
                # regardless of suffix case. xml:base can redirect local hrefs.
                for node in tree.iter():
                    for key,value in node.attrib.items():
                        key=key.rsplit('}',1)[-1]
                        if key=='base':raise Unsafe('XML base outside passive subset')
                        if not markup:
                            if key in ('srcset','poster','background','data'):raise Unsafe('XML resource attribute outside passive subset')
                            if key in ('href','src','full-path'):local_uri(value,False)
                if any(n.tag.rsplit('}',1)[-1]=='svg' for n in tree.iter()):svg(payload)
                if markup:
                    html(payload)
                if lower.endswith(".rels"):
                    for rel in tree:
                        target = rel.get("Target", "")
                        if rel.get("TargetMode") == "External":
                            if not rel.get("Type", "").endswith("/hyperlink"):
                                raise Unsafe("Relazione OOXML esterna attiva")
                            local_uri(target)
                        else:
                            relative(target)
            elif lower.endswith('.svg'):
                svg(payload)
            elif lower.endswith(".css"):
                css(payload)
            elif lower.endswith(".png"):
                from png_codec import decode
                decode(payload)
            elif lower!='mimetype':
                raise Unsafe('ZIP member outside finite passive subset: '+name)
            members[name] = payload
    return members


def png(data):
    import struct
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 33:
        raise Unsafe("PNG invalido")
    w, h = struct.unpack(">II", data[16:24])
    if w <= 0 or h <= 0 or w * h > MAX_PIXELS:
        raise Unsafe("Limite pixel PNG")
    return w, h


def tree_inventory(root):
    """Solo lstat; non discende symlink. Hash solo regular no-follow."""
    from data import digest_bytes
    root = no_symlink_root(root)
    pending, records = [root], []
    while pending:
        parent = pending.pop()
        for path in sorted(parent.iterdir()):
            info = path.lstat()
            name = path.relative_to(root).as_posix()
            row = {"path": name, "bytes": info.st_size, "mode": stat.S_IFMT(info.st_mode), "nlink": info.st_nlink}
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
                row["kind"] = "directory"
            elif stat.S_ISLNK(info.st_mode):
                row.update(kind="symlink", target=os.readlink(path))
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink!=1 or info.st_size>MAX_FILE:
                    row.update(kind='unsafe_regular',reason='hardlink or file size limit; target bytes not read')
                else:
                    row.update(kind="regular", sha256=digest_bytes(regular(root, name)))
            else:
                row["kind"] = "special"
            records.append(row)
    return sorted(records, key=lambda r: r["path"])
