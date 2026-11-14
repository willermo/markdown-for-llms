"""Readers of the twins: each turns a twin into the neutral model of `bench.model`.

Standard library only (`zipfile`, `xml.etree`, `html.parser`, `tarfile`): no library that
Marker uses (`openpyxl`, `python-pptx`, `mammoth`, `ebooklib`, ...), which a test enforces.
"""

from ..codes import BenchError, Code


def read_twin(doc, files, cache=None):
    """The twin of a document as a `model.Twin`, from the verified files of `fetch.resolve_document`."""
    twin = doc["twin"]
    kind = twin["kind"]
    if kind == "latex":
        from .latex import read_latex
        from .safe import read_tar
        members = read_tar(files.twin["file"], {twin["main"], twin["bbl"]})
        return read_latex(members[twin["main"]], members[twin["bbl"]])
    if kind == "jats":
        from .jats import read_jats
        return read_jats(files.twin["file"])
    if kind == "docx":
        from .docx import read_docx
        return read_docx(files.twin["file"])
    if kind == "pptx":
        from .pptx import read_pptx
        return read_pptx(files.twin["file"])
    if kind == "xlsx":
        from .xlsx import read_xlsx
        return read_xlsx(files.twin["file"])
    if kind == "xlsx-projection":
        from .xlsx import project
        from .. import tools
        layout = tools.run("pdftotext", ["-layout", tools.In(files.input), "-"], cache.tool_work_dir, timeout=300)
        return project(files.twin["file"], twin["select"], layout.decode("utf-8", "replace"))
    if kind == "odt":
        from .odt import read_odt
        return read_odt(files.twin["file"])
    if kind == "epub":
        from .epub import read_epub
        return read_epub(files.twin["file"])
    if kind == "wikisource":
        from .wikisource import read_wikisource
        return read_wikisource(files.twin["pages"], files.twin["quality"], twin.get("isolated_pages", ()))
    if kind == "akoma-ntoso":
        from .akn import read_akn
        return read_akn(files.twin["file"], tuple(twin.get("discard", ())))
    raise BenchError(Code.MANIFEST_INVALID, "twin kind " + kind)
