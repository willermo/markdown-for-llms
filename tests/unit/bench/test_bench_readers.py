"""Twin readers on minimal synthetic twins, and hostile archives and XML."""
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from markdown_for_llms.bench import selfcheck
from markdown_for_llms.bench.codes import BenchError, Code
from markdown_for_llms.bench.readers import safe
from markdown_for_llms.bench.readers.akn import read_akn
from markdown_for_llms.bench.readers.docx import read_docx
from markdown_for_llms.bench.readers.epub import read_epub
from markdown_for_llms.bench.readers.jats import read_jats
from markdown_for_llms.bench.readers.latex import read_latex
from markdown_for_llms.bench.readers.odt import read_odt
from markdown_for_llms.bench.readers.pptx import read_pptx
from markdown_for_llms.bench.readers.wikisource import read_wikisource
from markdown_for_llms.bench.readers.xlsx import project, read_xlsx

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'


def types(twin):
    return [e.type for e in twin.elements]


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def zip(self, name, members):
        path = self.root / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for member, data in members.items():
                z.writestr(member, data)
        return path

    def file(self, name, data):
        path = self.root / name
        path.write_bytes(data.encode() if isinstance(data, str) else data)
        return path


class DocxTests(Base):
    def test_docx(self):
        doc = f"""<w:document {W}><w:body>
<w:p><w:pPr><w:pStyle w:val="H1"/></w:pPr><w:r><w:t>Intro</w:t></w:r></w:p>
<w:p><w:r><w:t xml:space="preserve">Plain </w:t></w:r><w:r><w:rPr><w:b/></w:rPr><w:t>bold</w:t></w:r><w:r><w:rPr><w:i/></w:rPr><w:t>words</w:t></w:r><w:r><w:footnoteReference w:id="1"/></w:r><m:oMath><m:r><m:t>x</m:t></m:r></m:oMath></w:p>
<w:p><w:pPr><w:numPr><w:ilvl w:val="1"/></w:numPr></w:pPr><w:r><w:t>item</w:t></w:r></w:p>
<w:p><m:oMathPara><m:oMath><m:sSup><m:e><m:r><m:t>a</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup></m:oMath></m:oMathPara></w:p>
<w:tbl><w:tr><w:trPr><w:tblHeader/></w:trPr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr><w:p><w:r><w:t>H</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:tcPr><w:vMerge w:val="restart"/></w:tcPr><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc><w:tc><w:p><w:r><w:t>C</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
</w:body></w:document>"""
        styles = f'<w:styles {W}><w:style w:styleId="H1"><w:name w:val="heading 1"/></w:style></w:styles>'
        notes = f'<w:footnotes {W}><w:footnote w:type="separator" w:id="0"><w:p/></w:footnote><w:footnote w:id="1"><w:p><w:r><w:t>The note.</w:t></w:r></w:p></w:footnote></w:footnotes>'
        twin = read_docx(self.zip("t.docx", {"word/document.xml": doc, "word/styles.xml": styles, "word/footnotes.xml": notes}))
        self.assertEqual(types(twin), ["heading", "paragraph", "footnote", "list_item", "formula", "table"])
        h, p, n, li, f, t = twin.elements
        self.assertEqual((h.text, h.level), ("Intro", 1))
        self.assertEqual((p.text, p.math, p.emphasis), ("Plain boldwords", ("x",), (("b", "bold"), ("i", "words"))))
        self.assertEqual((n.text, li.level, f.text), ("The note.", 2, "a^2"))
        self.assertEqual([(c.text, c.row, c.col, c.rowspan, c.colspan, c.header) for c in t.cells],
                         [("H", 0, 0, 1, 2, True), ("A", 1, 0, 2, 1, False), ("B", 1, 1, 1, 1, False), ("C", 2, 1, 1, 1, False)])
        self.assertEqual(t.level, 1)


class PptxTests(Base):
    def test_pptx(self):
        a = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
        pres = f'<p:presentation {a}><p:sldIdLst><p:sldId id="256" r:id="rId2"/><p:sldId id="257" r:id="rId1"/></p:sldIdLst></p:presentation>'
        rels = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Target="slides/slide2.xml"/></Relationships>')
        def slide(body):
            return f'<p:sld {a}><p:cSld><p:spTree>{body}</p:spTree></p:cSld></p:sld>'
        title = ('<p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>First</a:t></a:r></a:p></p:txBody></p:sp>'
                 '<p:sp><p:nvSpPr><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:pPr><a:buChar char="•"/></a:pPr><a:r><a:rPr b="1"/><a:t>Point</a:t></a:r></a:p>'
                 '<a:p><a:fld type="slidenum"><a:t>2</a:t></a:fld></a:p></p:txBody></p:sp>'
                 '<p:pic/>')
        table = ('<p:sp><p:txBody><a:p><a:r><a:t>Second slide text</a:t></a:r><a:br/><a:r><a:t>line</a:t></a:r></a:p></p:txBody></p:sp>'
                 '<p:graphicFrame><a:graphic><a:graphicData><a:tbl><a:tr><a:tc gridSpan="2"><a:txBody><a:p><a:r><a:t>H</a:t></a:r></a:p></a:txBody></a:tc><a:tc hMerge="1"/></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>')
        twin = read_pptx(self.zip("t.pptx", {"ppt/presentation.xml": pres, "ppt/_rels/presentation.xml.rels": rels,
                                             "ppt/slides/slide1.xml": slide(table), "ppt/slides/slide2.xml": slide(title),
                                             "ppt/notesSlides/notesSlide1.xml": slide("<p:sp><p:txBody><a:p><a:r><a:t>speaker</a:t></a:r></a:p></p:txBody></p:sp>")}))
        self.assertEqual([(e.type, e.pos) for e in twin.elements],
                         [("heading", "pg1/h1"), ("list_item", "pg1/li1"), ("figure", "pg1/fig1"), ("paragraph", "pg2/p1"), ("table", "pg2/t1")])
        self.assertEqual(twin.elements[3].text, "Second slide text line")
        self.assertEqual(twin.elements[1].emphasis, (("b", "Point"),))
        self.assertEqual([(c.text, c.colspan) for c in twin.elements[4].cells], [("H", 2)])


S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


class XlsxTests(Base):
    def workbook(self):
        wb = f'<workbook {S}><sheets><sheet name="Prospetto_1" sheetId="1" r:id="rId1"/></sheets></workbook>'
        rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>'
        shared = f'<sst {S}><si><t>Titolo</t></si><si><t>Italia</t></si></sst>'
        styles = f'<styleSheet {S}><numFmts><numFmt numFmtId="164" formatCode="0.0"/></numFmts><cellXfs><xf numFmtId="0"/><xf numFmtId="164"/><xf numFmtId="3"/></cellXfs></styleSheet>'
        sheet = (f'<worksheet {S}><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row>'
                 '<row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2" s="1"><v>8.04</v></c><c r="C2" s="2"><v>1458</v></c></row>'
                 '<row r="3"><c r="A3" t="s"><v>1</v></c><c r="B3" s="1"><v>5</v></c><c r="C3" s="2"><v>99</v></c></row>'
                 '</sheetData><mergeCells><mergeCell ref="A1:C1"/></mergeCells></worksheet>')
        return self.zip("t.xlsx", {"xl/workbook.xml": wb, "xl/_rels/workbook.xml.rels": rels, "xl/sharedStrings.xml": shared,
                                   "xl/styles.xml": styles, "xl/worksheets/sheet1.xml": sheet})

    def test_xlsx_display_and_spans(self):
        twin = read_xlsx(self.workbook())
        self.assertEqual([(e.type, e.text) for e in twin.elements[:1]], [("heading", "Prospetto_1")])
        cells = twin.elements[1].cells
        self.assertEqual([(c.text, c.colspan, c.value, c.digits) for c in cells],
                         [("Titolo", 3, "", -1), ("Italia", 1, "", -1), ("8,0", 1, "8.04", 1), ("1.458", 1, "1458", 0),
                          ("Italia", 1, "", -1), ("5,0", 1, "5", 1), ("99", 1, "99", 0)])

    def test_projection_keeps_rows_found_once(self):
        layout = "Italia      8,0     1.458\nAltro 5,0 99\nripetuto 5,0 99\n"
        twin = project(self.workbook(), [{"sheet": "Prospetto_1", "rows": [2, 3], "cols": "B:C", "header_rows": [1]}], layout)
        self.assertEqual([(c.text, c.row, c.col) for c in twin.elements[0].cells], [("8,0", 0, 0), ("1.458", 0, 1)])
        self.assertEqual(twin.stats, {"projection_not_determinable": 1})
        with self.assertRaises(BenchError):
            project(self.workbook(), [{"sheet": "Missing", "rows": [2], "cols": "B:C", "header_rows": []}], layout)


class OdtTests(Base):
    def test_odt(self):
        ns = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
              'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
              'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"')
        content = f"""<office:document-content {ns}><office:automatic-styles><style:style style:name="T1"><style:text-properties fo:font-weight="bold"/></style:style></office:automatic-styles>
<office:body><office:text><text:table-of-content><text:index-body><text:p>Contents 3</text:p></text:index-body></text:table-of-content>
<text:h text:outline-level="2">Shapes</text:h><text:p>Some <text:span text:style-name="T1">bold</text:span> text<text:note><text:note-citation>1</text:note-citation><text:note-body><text:p>A note.</text:p></text:note-body></text:note><draw:frame><draw:image/></draw:frame></text:p>
<text:list><text:list-item><text:p>one</text:p><text:list><text:list-item><text:p>two</text:p></text:list-item></text:list></text:list-item></text:list>
<table:table><table:table-header-rows><table:table-row><table:table-cell table:number-columns-spanned="2"><text:p>H</text:p></table:table-cell><table:covered-table-cell/></table:table-row></table:table-header-rows>
<table:table-row><table:table-cell><text:p>a</text:p></table:table-cell><table:table-cell><text:p>b</text:p></table:table-cell></table:table-row></table:table>
</office:text></office:body></office:document-content>"""
        twin = read_odt(self.zip("t.odt", {"content.xml": content}))
        self.assertEqual(types(twin), ["heading", "paragraph", "figure", "footnote", "list_item", "list_item", "table"])
        self.assertEqual(twin.elements[1].emphasis, (("b", "bold"),))
        self.assertEqual([e.level for e in twin.elements[4:6]], [1, 2])
        self.assertEqual([(c.text, c.colspan, c.header) for c in twin.elements[6].cells], [("H", 2, True), ("a", 1, False), ("b", 1, False)])


class JatsTests(Base):
    def test_jats(self):
        xml = """<article xmlns:mml="http://www.w3.org/1998/Math/MathML"><front><article-meta><title-group><article-title>The title</article-title></title-group>
<abstract><p>Abstract text.</p></abstract></article-meta></front><body><sec><title>Methods</title>
<p>We use <italic>care</italic> and <inline-formula><mml:math><mml:mi>x</mml:mi></mml:math></inline-formula> here<xref ref-type="bibr">1</xref>.</p>
<disp-formula><mml:math><mml:msup><mml:mi>a</mml:mi><mml:mn>2</mml:mn></mml:msup></mml:math></disp-formula>
<table-wrap><label>Table 1</label><caption><p>Counts</p></caption><table><thead><tr><th>A</th><th>B</th></tr></thead>
<tbody><tr><td rowspan="2">x</td><td>1</td></tr><tr><td>2</td></tr></tbody></table><table-wrap-foot><fn><p>Foot.</p></fn></table-wrap-foot></table-wrap>
<fig><label>Fig. 1</label><caption><p>Picture</p></caption></fig></sec></body>
<back><ref-list><ref><mixed-citation>Smith J. Paper. 2020.</mixed-citation></ref><ref><element-citation><person-group><name><surname>Rossi</surname><given-names>A</given-names></name></person-group><article-title>Other</article-title><year>2021</year></element-citation></ref></ref-list></back></article>"""
        twin = read_jats(self.file("t.xml", xml))
        self.assertEqual(types(twin), ["heading", "heading", "paragraph", "heading", "paragraph", "formula", "paragraph", "table",
                                       "footnote", "figure", "bib_entry", "bib_entry"])
        p = twin.elements[4]
        self.assertEqual((p.text, p.math, p.emphasis), ("We use care and here1.", ("x",), (("i", "care"),)))
        self.assertEqual(twin.elements[5].text, "a^2")
        self.assertEqual([(c.text, c.row, c.col, c.rowspan, c.header) for c in twin.elements[7].cells],
                         [("A", 0, 0, 1, True), ("B", 0, 1, 1, True), ("x", 1, 0, 2, False), ("1", 1, 1, 1, False), ("2", 2, 1, 1, False)])
        self.assertEqual(twin.elements[9].text, "Fig. 1 Picture")
        self.assertEqual(twin.elements[11].text, "Rossi A Other 2021")


class EpubTests(Base):
    def test_epub_with_malformed_xhtml(self):
        container = ('<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                     '<rootfile full-path="OPS/content.opf"/></rootfiles></container>')
        opf = ('<package xmlns="http://www.idpf.org/2007/opf"><manifest><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
               '<item id="c1" href="text/c1.xhtml" media-type="application/xhtml+xml"/></manifest>'
               '<spine><itemref idref="nav"/><itemref idref="c1"/></spine></package>')
        page = """<html><body><nav>Contents</nav><h1>Chapter</h1><p>Text with <em>stress</em> and<a epub:type="noteref" href="#n1">1</a> more.
<!-- broken comment ---> <ul><li>one<ul><li>two</li></ul></li></ul>
<table><tr><th>A</th><th>B</th><tr><td colspan="2">wide</td></table>
<figure><img src="x.png"/><figcaption>A figure</figcaption></figure>
<section epub:type="footnotes"><ol><li id="n1"><p>The note. <a href="#r1">↩</a></p></li></ol></section></body></html>"""
        twin = read_epub(self.zip("t.epub", {"META-INF/container.xml": container, "OPS/content.opf": opf, "OPS/text/c1.xhtml": page,
                                             "OPS/nav.xhtml": "<html><body><p>Navigation only</p></body></html>"}))
        self.assertEqual(types(twin), ["heading", "paragraph", "list_item", "list_item", "table", "figure", "footnote"])
        self.assertEqual(twin.elements[1].text, "Text with stress and more.")
        self.assertEqual(twin.elements[1].emphasis, (("i", "stress"),))
        self.assertEqual([e.level for e in twin.elements[2:4]], [1, 2])
        self.assertEqual([(c.text, c.colspan, c.header) for c in twin.elements[4].cells], [("A", 1, True), ("B", 1, True), ("wide", 2, False)])
        self.assertEqual((twin.elements[5].text, twin.elements[6].text), ("A figure", "The note."))


class WikisourceTests(unittest.TestCase):
    def test_page(self):
        page = ("<noinclude><pagequality level=\"4\" />{{RigaIntestazione||— 199 —|}}</noinclude>{{nop}}\n"
                "''Corsivo'' e {{Sc|Stolz}} con {{AutoreCitato|Richard Dedekind|Dedekind}}<ref>Nota con ''titolo''.</ref> e {{Ec|scrito|scritto}}.\n\n"
                "Fine {{Pt|conti-|}}\n\n{{FI|file=x.png|caption=Fig. 1.}}\n\n{{Sconosciuto|testo perso}} visibile [[w:Pagina|collegamento]].\n"
                "{|\n! A !! B\n|-\n| colspan=\"2\" | wide\n|}\n<noinclude>{{PieDiPagina||12|}}</noinclude>")
        nxt = "{{Pt|nuo|continuo}} resto."
        twin = read_wikisource({1: {"wikitext": page}, 2: {"wikitext": nxt}, 3: {"wikitext": "skipped"}}, {1: 4, 2: 4, 3: 0})
        got = [(e.type, e.pos, e.text) for e in twin.elements if e.type != "table"]
        self.assertEqual(got, [("noise", "pg1/z1", "— 199 —"), ("noise", "pg1/z2", "12"),
                               ("paragraph", "pg1/p1", "Corsivo e Stolz con Dedekind e scrito."),
                               ("paragraph", "pg1/p2", "Fine"), ("figure", "pg1/fig1", "Fig. 1."),
                               ("paragraph", "pg1/p3", "visibile collegamento."), ("footnote", "pg1/n1", "Nota con titolo."),
                               ("paragraph", "pg2/p1", "continuo resto.")])
        self.assertEqual(twin.elements[2].emphasis, (("i", "Corsivo"),))
        table = next(e for e in twin.elements if e.type == "table")
        self.assertEqual([(c.text, c.colspan, c.header) for c in table.cells], [("A", 1, True), ("B", 1, True), ("wide", 2, False)])
        self.assertEqual(twin.stats["unknown_templates"], 1)
        self.assertEqual(twin.stats["pages_not_checked"], 1)


AKN = """<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"><act><preface><p><docType>DECRETO</docType> <docTitle>Titolo</docTitle></p>
<p><authorialNote><p>Entrata in vigore: oggi</p></authorialNote></p></preface>
<preamble><formula><p>IL PRESIDENTE</p></formula><citations><citation><p>Visto l'articolo</p></citation></citations></preamble>
<body><article><num>Art. 1.</num><heading>Oggetto</heading><paragraph><num>1.</num><list><intro><p>Si applica a:</p></intro>
<point><num>a)</num><content><p>primo punto</p></content></point></list></paragraph><section><content><p><authorialNote><p>NOTE dell'articolo</p></authorialNote></p></content></section></article></body>
<conclusions><p>Dato a Roma, addi' 8 ottobre</p></conclusions></act></akomaNtoso>"""


class AknTests(Base):
    def test_akn_and_declared_preface_rule(self):
        path = self.file("d.xml", AKN)
        twin = read_akn(path, ("normattiva-preface-notes",))
        self.assertEqual([(e.type, e.text) for e in twin.elements],
                         [("heading", "DECRETO Titolo"), ("paragraph", "IL PRESIDENTE"), ("paragraph", "Visto l'articolo"),
                          ("heading", "Art. 1. Oggetto"), ("paragraph", "1. Si applica a:"), ("list_item", "a) primo punto"),
                          ("footnote", "NOTE dell'articolo"), ("paragraph", "Dato a Roma, addi' 8 ottobre")])
        self.assertEqual(twin.stats, {"discarded_preface_notes": 1})
        without = read_akn(path, ())
        self.assertIn(("paragraph", "Entrata in vigore: oggi"), [(e.type, e.text) for e in without.elements])


TEX = r"""\documentclass{article}
\newcommand{\R}{\mathbb{R}}\newcommand{\pair}[1]{(#1,#1)}
\newtheorem{theorem}{Theorem}\newtheorem{lemma}[theorem]{Lemma}
\title{A \emph{small} paper}
\begin{document}\maketitle
\begin{abstract}We study $\R$. % a comment
\end{abstract}
\section{Intro}
Text with \emph{emphasis}, a citation \cite{k} and a note\footnote{Note text.} on $x\le y$ and \pair{z}.

\begin{equation}a^2+b^2\label{e}\end{equation}
\begin{theorem}First result.\end{theorem}\begin{lemma}[Key]Second.\end{lemma}
\begin{itemize}\item one \begin{enumerate}\item two\end{enumerate}\end{itemize}
\begin{figure}\includegraphics{x}\caption{A plot}\end{figure}
\begin{tabular}{cc}\hline \multicolumn{2}{c}{H}\\ a & b\\\end{tabular}
\begin{picture}(1,1)\put(0,0){x}\end{picture}
\bibliography{refs}
\end{document}"""
BBL = r"""\begin{thebibliography}{2}\providecommand{\natexlab}[1]{#1}
\bibitem{k} A.~Author, \emph{Title}, J. 1 (2020).
\bibitem[{B(2021)}]{b}\BibitemOpen\bibfield{author}{\bibinfo{author}{B. Bee}},\ \bibinfo{journal}{Phys.}\BibitemShut{NoStop}
\end{thebibliography}"""


class LatexTests(unittest.TestCase):
    def test_latex(self):
        twin = read_latex(TEX.encode(), BBL.encode())
        got = [(e.type, e.text) for e in twin.elements]
        self.assertEqual(got[:9], [("heading", "A small paper"), ("paragraph", "We study ."), ("heading", "Intro"),
                                   ("paragraph", "Text with emphasis, a citation and a note on and (z,z)."),
                                   ("footnote", "Note text."), ("formula", "a^2+b^2\\label{e}"),
                                   ("paragraph", "Theorem 1. First result."), ("paragraph", "Lemma 2 (Key). Second."),
                                   ("list_item", "one")])
        self.assertEqual(got[9:], [("list_item", "two"), ("figure", "A plot"), ("table", ""), ("bib_entry", "A. Author, Title, J. 1 (2020)."),
                                   ("bib_entry", "B. Bee, Phys.")])
        self.assertEqual(twin.elements[1].math, ("\\mathbb{R}",))
        self.assertEqual(twin.elements[3].math, ("x\\le y",))
        self.assertEqual(twin.elements[3].emphasis, (("i", "emphasis"),))
        self.assertEqual([e.level for e in twin.elements if e.type == "list_item"], [1, 2])
        self.assertEqual([(c.text, c.colspan) for c in twin.elements[11].cells], [("H", 2), ("a", 1), ("b", 1)])
        self.assertEqual(twin.stats, {"skipped_picture": 1})


class HostileInputTests(Base):
    def test_zip_traversal_and_bomb(self):
        evil = self.zip("evil.zip", {"../evil.xml": "<a/>"})
        with self.assertRaises(BenchError) as ctx:
            safe.SafeZip(evil)
        self.assertEqual(ctx.exception.code, Code.ARCHIVE_UNSAFE)
        bomb = self.zip("bomb.docx", {"word/document.xml": "0" * (8 * 1024 * 1024)})
        with self.assertRaises(BenchError) as ctx:
            read_docx(bomb)
        self.assertEqual(ctx.exception.code, Code.ARCHIVE_UNSAFE)

    def test_xml_entities_and_internal_subset(self):
        laughs = '<?xml version="1.0"?><!DOCTYPE l [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]><l>&b;</l>'
        external = '<!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]><x>&e;</x>'
        for data in (laughs, external):
            with self.subTest(data=data[:30]), self.assertRaises(BenchError) as ctx:
                read_jats(self.file("x.xml", data))
            self.assertEqual(ctx.exception.code, Code.XML_UNSAFE)
        with self.assertRaises(BenchError) as ctx:
            read_jats(self.file("bad.xml", "<a><b></a>"))
        self.assertEqual(ctx.exception.code, Code.TWIN_UNREADABLE)

    def test_flat_odp_of_the_self_check_is_parsed_safely(self):
        """The flat ODP that LibreOffice writes for the PPTX self-check goes through `parse_xml` too (K5)."""
        laughs = (b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]>'
                  b'<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0">&b;</office:document>')
        with self.assertRaises(BenchError) as ctx:
            selfcheck.fodp_inventory(laughs)
        self.assertEqual(ctx.exception.code, Code.XML_UNSAFE)
        with self.assertRaises(BenchError) as ctx:
            selfcheck.fodp_inventory(b"<a><b></a>")
        self.assertEqual(ctx.exception.code, Code.TWIN_UNREADABLE)

    def test_tar_link_member(self):
        path = self.root / "src.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            info = tarfile.TarInfo("main.tex")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        with self.assertRaises(BenchError) as ctx:
            safe.read_tar(path, {"main.tex"})
        self.assertEqual(ctx.exception.code, Code.ARCHIVE_UNSAFE)
        with tarfile.open(path, "w:gz") as tar:
            data = b"x"
            info = tarfile.TarInfo("../main.tex")
            info.size = 1
            tar.addfile(info, io.BytesIO(data))
        with self.assertRaises(BenchError):
            safe.read_tar(path, {"main.tex"})
