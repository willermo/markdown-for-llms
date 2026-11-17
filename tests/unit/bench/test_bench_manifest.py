"""The packaged manifest of the public reference set and its strict validator."""
import copy
import json
import unittest

from markdown_for_llms.bench import manifest
from markdown_for_llms.bench.codes import BenchError, Code


LAST = 36
QUICK = ["D04", "D08", "D15", "D18", "D19", "D20", "D35"]
MISSING_FORMATS = set()
REFERENCES = {"D14": "684d194e", "D17": "2c611573", "D18": "586e4ff6", "D19": "cdbf7f4b", "D22": "da5df196",
              "D23": "25e315d9", "D24": "9c45455d", "D25": "0c84c18e", "D27": "de3fb983", "D28": "4b72bc84",
              "D29": "cdd58709", "D30": "b86797ce", "D31": "04e0e3c3", "D32": "879db963", "D33": "70fb9eda",
              "D34": "111cf0ab", "D35": "d5d393f5", "D36": "1e52c541"}


def raw(data):
    return json.dumps(data).encode()


class PackagedManifestTests(unittest.TestCase):
    def setUp(self):
        self.m = manifest.load()

    def test_documents_levels_and_profiles(self):
        ids = [d["id"] for d in self.m.documents]
        self.assertEqual(self.m.data["schema"], "mdf-bench-manifest/2")
        self.assertEqual(ids, [f"D{n:02d}" for n in range(1, LAST + 1)])
        self.assertEqual([d["id"] for d in self.m.select("quick")], QUICK)
        self.assertEqual([d["id"] for d in self.m.select("long")], ["D17", "D21"])
        self.assertEqual([d["id"] for d in self.m.select("standard")], [i for i in ids if i not in ("D17", "D21")])
        self.assertEqual({d["profile"] for d in self.m.documents}, set(manifest.PROFILES))
        self.assertEqual([d["id"] for d in self.m.select(ids=["D19", "D04", "D19"])], ["D04", "D19"])
        with self.assertRaises(BenchError) as ctx:
            self.m.select(ids=["D99"])
        self.assertEqual(ctx.exception.code, Code.UNKNOWN_DOCUMENT)

    def test_formats_languages_and_scripts(self):
        formats = {d["format"] for d in self.m.documents}
        self.assertEqual(formats, set(manifest.FORMATS) - MISSING_FORMATS)
        by = {d["id"]: (d["language"], d["scripts"]) for d in self.m.documents}
        self.assertEqual(by["D15"], (["uk", "en"], ["Cyrl", "Latn"]))
        self.assertEqual(by["D30"], (["uk", "en"], ["Cyrl", "Latn"]))
        self.assertEqual(by["D19"], (["it"], ["Latn"]))
        self.assertEqual(by["D26"], (["en"], ["Latn"]))
        self.assertEqual(self.m.document("D17")["limits"], {"max_pages": 300, "output_mib": 1024})
        self.assertEqual(self.m.document("D21")["limits"], {"max_pages": 300, "output_mib": 1024})

    def test_txt_inputs_and_their_oracles(self):
        txt = [d for d in self.m.documents if d["format"] == "txt"]
        for d in txt:
            with self.subTest(doc=d["id"]):
                recode = [s for s in d["input"]["derive"] if s["op"] == "recode"][-1]
                self.assertEqual(d["expected_encoding"], manifest.ENCODINGS[recode["encoding"]])
                self.assertRegex(d["input"]["reference"]["oracle_sha256"], "^[0-9a-f]{64}$")
        # D29 (UTF-8) and D30 (UTF-16) are the same text: the same oracle
        self.assertEqual(self.m.document("D29")["input"]["reference"]["oracle_sha256"],
                         self.m.document("D30")["input"]["reference"]["oracle_sha256"])
        self.assertEqual({d["id"]: d["expected_encoding"] for d in txt},
                         {"D29": "utf_8", "D30": "utf_16", "D31": "cp1252", "D32": "cp1250", "D33": "cp1251",
                          "D34": "cp1253", "D35": "cp1252", "D36": "cp1251"})
        self.assertEqual([d["id"] for d in txt if d["twin"] is None], ["D35", "D36"])
        self.assertTrue(all(d.get("checks") == "encoding" for d in txt if d["twin"] is None))

    def test_gutenberg_sources_from_the_foundation_mirror(self):
        # www.gutenberg.org was down for maintenance on 16/11; Davide chose the PGLAF mirror (17:19)
        pg = {k: v for k, v in self.m.sources.items() if k[:3] in ("d31", "d32", "d33", "d34")}
        self.assertEqual(len(pg), 8)
        self.assertTrue(all(v["url"].startswith("https://gutenberg.pglaf.org/cache/epub/") for v in pg.values()))
        self.assertEqual({self.m.document(f"D3{k}")["language"][0] for k in range(1, 5)}, {"it", "cs", "ru", "el"})
        self.assertEqual(self.m.document("D33")["scripts"], ["Cyrl"])

    def test_every_file_source_is_used_and_hashed(self):
        used = set()
        for d in self.m.documents:
            used.add(d["input"]["source"])
            if d["twin"]:
                used.add(d["twin"]["source"])
            if d["twin"] and "inventory" in d["twin"]:
                used.add(d["twin"]["inventory"])
        self.assertEqual(used, set(self.m.sources))
        self.assertEqual(len(self.m.sha256), 64)

    def test_wikisource_revisions(self):
        d14 = self.m.sources["d14-wikisource"]["pages"]
        self.assertEqual([p["page"] for p in d14], list(range(1, 13)))
        self.assertTrue(all(p["quality"] == 4 for p in d14))
        self.assertEqual(d14[2]["revid"], 2021757)
        d17 = self.m.sources["d17-wikisource"]["pages"]
        self.assertEqual(len(d17), 272)
        self.assertEqual([p["page"] for p in d17 if p["quality"] != 4], [10, 117, 120, 272])
        self.assertEqual(self.m.document("D17")["twin"]["isolated_pages"], [10, 117, 120, 272])

    def test_paged_formats(self):
        for d in self.m.documents:
            with self.subTest(doc=d["id"]):
                if d["format"] in manifest.PAGED_FORMATS:
                    self.assertIsInstance(d["pages"], int)
                else:
                    self.assertIsNone(d["pages"])
                    self.assertGreater(d["pages_estimate"], 0)
        self.assertEqual(self.m.document("D19")["pages"], 7)
        self.assertEqual(self.m.document("D18")["input"]["derive"][-1],
                         {"op": "page-png", "page": 3, "dpi": 300, "color": "rgb"})

    def test_normattiva_sources_need_a_session(self):
        self.assertIn("session_url", self.m.sources["d19-akn"])
        html = self.m.sources["d19-normattiva-html"]
        self.assertIn("tipoDettaglio=originario", html["session_url"])
        self.assertEqual(html["sha256"], "0842830f705bcb8c81d2c7dfa4d8268e9d6b7441de779b47001a563bafbd3dfd")


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.data = copy.deepcopy(manifest.load().data)

    def rejects(self, data, detail=None):
        with self.assertRaises(BenchError) as ctx:
            manifest.parse(raw(data))
        self.assertEqual(ctx.exception.code, Code.MANIFEST_INVALID)
        if detail:
            self.assertIn(detail, ctx.exception.detail)

    def mutate(self, fn):
        data = copy.deepcopy(self.data)
        fn(data)
        return data

    def test_round_trip(self):
        self.assertEqual(manifest.parse(raw(self.data)).data, self.data)

    def test_rejections(self):
        cases = {
            "unknown top key": lambda d: d.update(extra=1),
            "personal set": lambda d: d.update(set="personal"),
            "unknown document key": lambda d: d["documents"][0].update(text="x"),
            "short sha256": lambda d: d["sources"]["d01-pdf"].update(sha256="ab" * 31),
            "upper-case sha256": lambda d: d["sources"]["d01-pdf"].update(sha256="AB" * 32),
            "http url": lambda d: d["sources"]["d01-pdf"].update(url="http://arxiv.org/pdf/x"),
            "unknown source": lambda d: d["documents"][0]["input"].update(source="nope"),
            "twin kind": lambda d: d["documents"][0]["twin"].update(kind="rtf"),
            "pages on docx": lambda d: d["documents"][9].update(pages=10),
            "duplicate id": lambda d: d["documents"][1].update(id="D01"),
            "level": lambda d: d["documents"][0].update(levels=["fast"]),
            "derive op": lambda d: d["documents"][13]["input"]["derive"].append({"op": "shell"}),
            "wikisource page order": lambda d: d["sources"]["d14-wikisource"]["pages"].reverse(),
            "twin pages outside index": lambda d: d["documents"][17]["twin"].update(pages=[99]),
            "bad sha1": lambda d: d["sources"]["d14-wikisource"]["pages"][0].update(sha1="x" * 40),
            # E7: the reference names the toolchain that produced it, not tool versions
            "reference with tool versions": lambda d: d["documents"][13]["input"]["reference"].update(tools={"ddjvu": "3.5.28"}),
            "reference without toolchain": lambda d: d["documents"][13]["input"]["reference"].pop("toolchain"),
            "reference toolchain not a sha256": lambda d: d["documents"][13]["input"]["reference"].update(toolchain="3.5.28"),
            # schema 2
            "no language": lambda d: d["documents"][0].pop("language"),
            "language not BCP 47": lambda d: d["documents"][0].update(language=["Italian"]),
            "script not ISO 15924": lambda d: d["documents"][0].update(scripts=["latin"]),
            "unknown limit": lambda d: d["documents"][0].update(limits={"pages": 3}),
            "twin null without the encoding checks": lambda d: d["documents"][28].update(twin=None),
            "encoding checks with a twin": lambda d: d["documents"][28].update(checks="encoding"),
            "encoding checks on a pdf": lambda d: d["documents"][0].update(checks="encoding", twin=None),
            "expected encoding not of the recode": lambda d: d["documents"][28].update(expected_encoding="utf_16"),
            "expected encoding on a pdf": lambda d: d["documents"][0].update(expected_encoding="utf_8"),
            "txt without oracle": lambda d: d["documents"][28]["input"]["reference"].pop("oracle_sha256"),
            "oracle without recode": lambda d: d["documents"][21]["input"]["reference"].update(oracle_sha256="0" * 64),
            "utf-16 without a mark": lambda d: d["documents"][29]["input"]["derive"][1].update(bom=False),
            "code page with a mark": lambda d: d["documents"][28]["input"]["derive"][1].update(encoding="cp1251", bom=True),
            "unknown encoding": lambda d: d["documents"][28]["input"]["derive"][1].update(encoding="latin-9"),
            "step kinds do not chain": lambda d: d["documents"][21]["input"]["derive"].append({"op": "page-png", "page": 1,
                                                                                               "dpi": 72, "color": "rgb"}),
            "last step not the format": lambda d: d["documents"][21]["input"]["derive"][0].update(to="rtf"),
            "office target": lambda d: d["documents"][21]["input"]["derive"][0].update(to="pdf"),
            "deterministic not a boolean": lambda d: d["documents"][22]["input"]["reference"].update(deterministic="no"),
            "excerpt backwards": lambda d: d["documents"][28]["input"]["derive"].insert(1, {"op": "excerpt", "first_line": 5,
                                                                                            "last_line": 2}),
            "image format": lambda d: d["documents"][26]["input"]["derive"][1].update(format="gif"),
        }
        for name, fn in cases.items():
            with self.subTest(case=name):
                self.rejects(self.mutate(fn))

    def test_schema_1_is_refused_with_its_own_code(self):
        for schema in ("mdf-bench-manifest/1", "mdf-bench-manifest/3"):
            with self.subTest(schema=schema):
                with self.assertRaises(BenchError) as ctx:
                    manifest.parse(raw(self.mutate(lambda d: d.update(schema=schema))))
                self.assertEqual(ctx.exception.code, Code.MANIFEST_SCHEMA_UNSUPPORTED)

    def test_duplicate_key_and_bad_json(self):
        with self.assertRaises(BenchError) as ctx:
            manifest.parse(b'{"schema": 1, "schema": 2}')
        self.assertIn("duplicate key", ctx.exception.detail)
        with self.assertRaises(BenchError):
            manifest.parse(b"\xff")


class ReferenceToolchainTests(unittest.TestCase):
    def test_references_were_produced_by_the_packaged_toolchain(self):
        from markdown_for_llms.bench import toolchain
        m = manifest.load()
        refs = {d["id"]: d["input"]["reference"] for d in m.documents if "derive" in d["input"]}
        self.assertEqual({r["toolchain"] for r in refs.values()}, {toolchain.identity()})
        # the derived inputs of the reference toolchain (the first four: proof 1 of the toolchain run; the
        # others: the first derivation of M6 of MFL026, deterministic but for D23 and D24)
        self.assertEqual({k: r["sha256"][:8] for k, r in refs.items()}, REFERENCES)
        self.assertEqual({k for k, r in refs.items() if r.get("deterministic") is False}, {"D23", "D24"})
