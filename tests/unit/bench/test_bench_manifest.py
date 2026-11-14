"""The packaged manifest of the public reference set and its strict validator."""
import copy
import json
import unittest

from markdown_for_llms.bench import manifest
from markdown_for_llms.bench.codes import BenchError, Code


def raw(data):
    return json.dumps(data).encode()


class PackagedManifestTests(unittest.TestCase):
    def setUp(self):
        self.m = manifest.load()

    def test_documents_levels_and_profiles(self):
        ids = [d["id"] for d in self.m.documents]
        self.assertEqual(ids, [f"D{n:02d}" for n in range(1, 20)])
        self.assertEqual([d["id"] for d in self.m.select("quick")], ["D04", "D15", "D18", "D19"])
        self.assertEqual([d["id"] for d in self.m.select("long")], ["D17"])
        self.assertEqual([d["id"] for d in self.m.select("standard")], [i for i in ids if i != "D17"])
        self.assertEqual({d["profile"] for d in self.m.documents}, set(manifest.PROFILES))
        self.assertEqual([d["id"] for d in self.m.select(ids=["D19", "D04", "D19"])], ["D04", "D19"])
        with self.assertRaises(BenchError) as ctx:
            self.m.select(ids=["D20"])
        self.assertEqual(ctx.exception.code, Code.UNKNOWN_DOCUMENT)

    def test_every_file_source_is_used_and_hashed(self):
        used = set()
        for d in self.m.documents:
            used.add(d["input"]["source"])
            used.add(d["twin"]["source"])
            if "inventory" in d["twin"]:
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
                if d["format"] in ("pdf", "png"):
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
            "schema": lambda d: d.update(schema="mdf-bench-manifest/2"),
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
        }
        for name, fn in cases.items():
            with self.subTest(case=name):
                self.rejects(self.mutate(fn))

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
        self.assertEqual(set(refs), {"D14", "D17", "D18", "D19"})
        self.assertEqual({r["toolchain"] for r in refs.values()}, {toolchain.identity()})
        # the four derived inputs of the reference toolchain (proof 1 of the toolchain run)
        self.assertEqual({k: r["sha256"][:8] for k, r in refs.items()},
                         {"D14": "684d194e", "D17": "2c611573", "D18": "586e4ff6", "D19": "cdbf7f4b"})
