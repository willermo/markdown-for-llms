"""Helpers of the tests of `mdf bench run`: a synthetic public set in a cache, a controlled engine,
and the bench run in a subprocess.

The set needs no toolchain: three txt inputs derived by `recode` from one UTF-8 text, two judged
on an HTML twin (a Project Gutenberg page with its boilerplate) and one on its oracle alone; the
engine converts them with its real `text` route. The toolchain image and the weights are not
verified (`fake_identity`), and the lock and the process check of the venv use a directory of
the test, never the development venv (the full check holds that one).

As a script, `python -I -B bench_run_support.py ROOT MODE BASE -- ARGS` runs `mdf bench run ARGS`
from the sources under ROOT with that identity, the venv `BASE/venv` and an engine that converts
(`ok`), stops at its first step until a signal comes after writing `BASE/started` (`wait`), or
publishes its result and then dies by SIGKILL before the bench records it (`kill`); with a
Datalab without network (`FakeDatalab`, its calls counted in `BASE/datalab.log`) that answers
(`datalab`) or whose first GET waits until a signal comes (`datalab-hold`).

`build_datalab_set` adds PDF documents to the set (D05, two pages; D06, one page), judged on the
same HTML twin; the PDF preparation of the Datalab route runs for real in its light child.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PARAGRAPHS = ["Il primo capitolo racconta di un falegname che trovò un pezzo di legno speciale.",
              "Il secondo capitolo descrive il pezzo di legno che piangeva e rideva come un bambino.",
              "Nel terzo capitolo il vecchio Geppetto prende il legno e comincia a fabbricare il burattino."]
TEXT = "\n\n".join(PARAGRAPHS) + "\n"
HTML = ("<html><head><title>t</title></head><body>"
        "<section class=\"pg-boilerplate\" id=\"pg-header\"><p>The Project Gutenberg eBook header licence.</p></section>"
        "<h2>Capitolo primo</h2>" + "".join(f"<p>{p}</p>" for p in PARAGRAPHS) +
        "<section class=\"pg-boilerplate\" id=\"pg-footer\"><p>End of the Project Gutenberg licence.</p></section>"
        "</body></html>")
FAKE_IMAGE = "sha256:" + "f" * 64


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build_set(base, text=TEXT, html=HTML):
    """A manifest, a cache and a runs directory under `base`: {manifest, cache, runs, venv}."""
    from markdown_for_llms.bench.derive import oracle_text, recode
    base = Path(base)
    cache, runs, venv = base / "cache", base / "runs", base / "venv"
    (cache / "sources").mkdir(parents=True)
    (venv / "bin").mkdir(parents=True)
    src, twin = text.encode("utf-8"), html.encode("utf-8")
    for data in (src, twin):
        (cache / "sources" / sha(data)).write_bytes(data)
    oracle = sha(oracle_text(text).encode("utf-8"))

    def doc(doc_id, encoding, with_twin, levels=("quick",)):
        d = {"id": doc_id, "title": "Synthetic " + doc_id, "role": "test", "license": "CC0", "attribution": "test",
             "levels": list(levels), "profile": "office", "format": "txt", "pages": None, "pages_estimate": 1,
             "language": ["it"], "scripts": ["Latn"], "expected_encoding": encoding.replace("-", "_"),
             "input": {"source": "t-src", "derive": [{"op": "recode", "encoding": encoding, "bom": False}],
                       "reference": {"sha256": sha(recode(text, encoding, False)), "oracle_sha256": oracle,
                                     "toolchain": "0" * 64}},
             "twin": {"kind": "html", "source": "t-html"} if with_twin else None}
        if not with_twin:
            d["checks"] = "encoding"
        return d
    manifest = {"schema": "mdf-bench-manifest/2", "set": "public",
                "sources": {"t-src": {"kind": "file", "url": "https://example.invalid/t.txt", "bytes": len(src),
                                      "sha256": sha(src)},
                            "t-html": {"kind": "file", "url": "https://example.invalid/t.html", "bytes": len(twin),
                                       "sha256": sha(twin)}},
                "documents": [doc("D01", "utf-8", True), doc("D02", "cp1252", True), doc("D03", "cp1252", False)]}
    path = base / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return {"manifest": path, "cache": cache, "runs": runs, "venv": venv, "base": base}


def fake_identity(model_root, image=FAKE_IMAGE):
    from markdown_for_llms.bench.adapters import engine as adapter
    from markdown_for_llms.engine import toolchain
    return {"package": adapter.package_identity(),
            "toolchain": {"content_sha256": toolchain.identity(), "image_id": image, "engine": "test", "code": None},
            "weights": {"manifest_sha256": "0" * 64, "files": 18, "bytes": 1, "code": None}}


class Engine:
    """The real engine with a controlled observer, called before the observer of the bench;
    `calls` records every conversion, `hooks` maps a step to a function called at that step
    (with the number of the call)."""

    def __init__(self, mode="ok", base=None, hooks=None):
        self.mode, self.base, self.calls, self.hooks = mode, base, [], dict(hooks or {})

    def __call__(self, source, out, request, limits, work, observer=None, name=None, model_root=None,
                 datalab_key=None):
        from markdown_for_llms.engine import core
        self.calls.append({"attempt": request["attempt"], "limits": dict(limits), "request": dict(request),
                           "key": datalab_key is not None})

        def chained(event):
            mine = self.observer(event)
            theirs = observer(event) if observer is not None else None
            return mine or theirs
        result = core.convert(source, out, request, limits, work, observer=chained, name=name,
                              model_root=model_root, datalab_key=datalab_key)
        if self.mode == "kill":
            os.kill(os.getpid(), signal.SIGKILL)
        if self.mode == "raise":
            raise Died()
        return result

    def observer(self, event):
        if self.mode == "wait" and event.get("event") == "step":
            (Path(self.base) / "started").write_text(str(os.getpid()))
            while True:
                time.sleep(0.05)
        hook = self.hooks.get(event.get("step")) if event.get("event") == "step" else None
        if hook is not None:
            hook(len(self.calls))
        return None


KEY = "dl-TEST-KEY-7c1e9a0b5d3f2468"


def write_key(base, mode=0o600, text=None):
    """A key file of the user with DATALAB_API_KEY (mode 0600 unless said otherwise)."""
    path = Path(base) / "datalab.env"
    path.write_text(text if text is not None else f"DATALAB_API_KEY={KEY}\n")
    os.chmod(path, mode)
    return path


class FakeDatalab:
    """Datalab without network. Every POST, GET and DELETE is a line of `BASE/datalab.log` (shared
    by the processes of a test); `post` answers the POST (default: accepted with a new request id),
    `answer` the GET that completes (default: the paragraphs of the set, billed `billed` cents);
    with `hold` the first GET writes `BASE/holding` and waits until a signal comes."""

    def __init__(self, base, post=None, answer=None, billed=0.5, hold=False):
        self.base, self.post_answer, self.answer, self.billed, self.hold = Path(base), post, answer, billed, hold

    def log(self, *fields):
        with open(self.base / "datalab.log", "a") as fh:
            fh.write(" ".join(map(str, fields)) + f" {time.monotonic():.3f}\n")

    def network_errors(self):
        return (OSError,)

    def post(self, url, headers, files, data):
        assert headers == {"X-API-Key": KEY}
        files["file"][1].read()
        self.log("POST", data["mode"])
        if self.post_answer is not None:
            return self.post_answer
        return 200, {"success": True, "request_id": "req-" + os.urandom(4).hex()}

    def get(self, url, headers):
        self.log("GET")
        if self.hold:
            (self.base / "holding").write_text(str(os.getpid()))
            while True:
                time.sleep(0.05)
        if self.answer is not None:
            return 200, self.answer
        breakdown = {} if self.billed is None else {"list_cost_cents": 1.0, "final_cost_cents": self.billed}
        return 200, {"status": "complete", "success": True, "markdown": "\n\n".join(PARAGRAPHS) + "\n",
                     "images": {}, "cost_breakdown": breakdown, "versions": {"marker": "fake"}}

    def delete(self, url, headers):
        self.log("DELETE")
        return 200, {"success": True}

    @staticmethod
    def calls(base, method):
        path = Path(base) / "datalab.log"
        return [line.split() for line in path.read_text().splitlines() if line.startswith(method + " ")] \
            if path.exists() else []


def build_datalab_set(base, text=TEXT, html=HTML):
    """The set of `build_set` with two PDF documents in the quick level: D05 (two pages) and D06 (one
    page), the paragraphs as native text, judged on the same HTML twin."""
    sys.path.insert(0, str(ROOT / "tests" / "unit" / "engine"))
    from engine_test_support import make_pdf
    s = build_set(base, text, html)
    data = json.loads(s["manifest"].read_text())

    def page(lines):
        return {"content": b"BT /F1 11 Tf 40 780 Td 16 TL " + b" ".join(b"(" + x.encode("latin-1") + b") '" for x in lines)
                + b" ET"}
    for doc_id, pages in (("D05", [PARAGRAPHS[:2], PARAGRAPHS[2:]]), ("D06", [PARAGRAPHS])):
        pdf = make_pdf([page(lines) for lines in pages])
        (s["cache"] / "sources" / sha(pdf)).write_bytes(pdf)
        data["sources"]["t-" + doc_id.lower()] = {"kind": "file", "url": f"https://example.invalid/{doc_id}.pdf",
                                                  "bytes": len(pdf), "sha256": sha(pdf)}
        data["documents"].append({"id": doc_id, "title": "Synthetic " + doc_id, "role": "test", "license": "CC0",
                                  "attribution": "test", "levels": ["quick"], "profile": "scientific", "format": "pdf",
                                  "pages": len(pages), "language": ["it"], "scripts": ["Latn"],
                                  "input": {"source": "t-" + doc_id.lower()}, "twin": {"kind": "html", "source": "t-html"}})
    s["manifest"].write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return s


class Died(BaseException):
    """The bench dies after the engine published its result (T1 in the same process)."""


def patches(venv, engine, identity=fake_identity, transport=None):
    """The patches of a run in this process: [(object, attribute, value)]; with `transport`, Datalab
    answers through it, without network, and its sends are DATALAB_SPACING_TEST seconds apart."""
    from markdown_for_llms.bench import run
    from markdown_for_llms.bench.adapters import engine as adapter
    from markdown_for_llms.engine import datalab
    out = [(adapter, "convert", engine), (adapter, "identity", identity), (run, "venv_root", lambda: Path(venv))]
    if transport is not None:
        out += [(datalab, "Transport", lambda: transport), (datalab, "POLL_SECONDS", 0.01),
                (run, "DATALAB_SPACING", DATALAB_SPACING_TEST)]
    return out


DATALAB_SPACING_TEST = 0.2


def spawn(mode, base, args, shell_background=False):
    """The bench run in a subprocess (Popen). With `shell_background`, a non-interactive shell starts
    it in the background (`&`), so it inherits SIGINT ignored, and writes its PID to BASE/pid."""
    argv = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), str(ROOT), mode, str(base), "--", *map(str, args)]
    if shell_background:
        script = " ".join(_quote(a) for a in argv) + f" & echo $! > {_quote(str(Path(base) / 'pid'))}; wait $!"
        return subprocess.Popen(["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _quote(text):
    return "'" + text.replace("'", "'\\''") + "'"


def wait_for(path, proc, timeout=60):
    """Wait until `path` exists while `proc` is alive; False when it ends first or time runs out."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if Path(path).exists():
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def tree_digest(run_dir):
    """SHA-256 of the state, of the list of attempts and of the ledger, to prove a run untouched."""
    run_dir = Path(run_dir)
    state = run_dir / "state.json"
    attempts = sorted(p.name for p in (run_dir / "attempts").iterdir()) if (run_dir / "attempts").exists() else []
    ledger = run_dir.parent / "spend.jsonl"
    return (sha(state.read_bytes()) if state.exists() else None, sha(json.dumps(attempts).encode()),
            sha(ledger.read_bytes()) if ledger.exists() else None)


def _main(argv):
    root, mode, base = argv[1], argv[2], Path(argv[3])
    sys.path.insert(0, root)
    args = argv[argv.index("--") + 1:]
    from markdown_for_llms.bench import cli
    transport = FakeDatalab(base, hold=mode == "datalab-hold") if mode.startswith("datalab") else None
    for obj, name, value in patches(base / "venv", Engine("ok" if transport else mode, base), transport=transport):
        setattr(obj, name, value)
    return cli.main(["run", *args])


if __name__ == "__main__":
    sys.exit(_main(sys.argv))


def make_docx(path, paragraphs):
    """A minimal docx with one paragraph per text (a twin of the personal set)."""
    import zipfile
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    rel = "http://schemas.openxmlformats.org/package/2006/relationships"
    body = "".join(f"<w:p><w:r><w:t>{t}</w:t></w:r></w:p>" for t in paragraphs)
    members = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/'
                               'package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.'
                               'openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="'
                               'application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.'
                               'openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        "_rels/.rels": f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{rel}"><Relationship Id="rId1" '
                       'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                       'Target="word/document.xml"/></Relationships>',
        "word/document.xml": f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{w}"><w:body>{body}</w:body>'
                             '</w:document>'}
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return Path(path)
