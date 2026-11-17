"""The StarMath channel (delta r3 Δ4.4, Δ4.7): the atomizer is frozen on examples in the style of
the LibreOffice Math command reference (never on D26), and the Flat ODT walker localizes the
formulas in their paragraphs and leaves indexes out."""
import unittest

from markdown_for_llms.bench import starmath
from markdown_for_llms.bench.match import FORMULA_RATIO
from markdown_for_llms.bench.readers.mathml import mathml_linear
from markdown_for_llms.bench.readers.safe import parse_xml
from markdown_for_llms.bench.text import formula_similarity, unicode_atoms

EXAMPLES = {
    "sum from{i=1} to{n} i^2": "∑ _ i = 1 ^ n i ^ 2",
    "sqrt{a^2 + b^2}": "√ a ^ 2 + b ^ 2",
    "nroot{3}{x}": "√ x 3",
    "x = {-b +- sqrt{b^2 - 4ac}} over {2a}": "x = - b ± √ b ^ 2 - 4 a c 2 a",
    "abs{x}": "| x |",
    "%alpha + %BETA = %iGAMMA": "α + Β = Γ",
    "lim from{x toward 0} {sin x} over x = 1": "l i m _ x → 0 s i n x x = 1",
    "int from a to b f(x) dx": "∫ _ a ^ b f ( x ) d x",
    "left( a over b right)": "( a b )",
    "left lbrace x right none": "{ x",
    "vec v cdot bar x": "v ^ ⃗ · x ^ ¯",
    '"text" cdot n': "t e x t · n",
    "color red x + size 16 y + bold z": "x + y + z",
    "matrix{a # b ## c # d}": "a b c d",
    "a <= b <> c approx d": "a ≤ b ≠ c ≈ d",
    "A subseteq B, x in setR": "A ⊆ B , x ∈ ℝ",
    "fact n": "n !",
    "log_{10} x": "l o g _ 1 0 x",
    "func f(x) = x sub 1 sup 2": "f ( x ) = x _ 1 ^ 2",
    "a_{i,j}^2": "a _ i , j ^ 2",
    "%PI %perthousand %infinite": "Π ‰ ∞",
}


class AtomizerTests(unittest.TestCase):
    def test_reference_examples(self):
        self.assertEqual(starmath.VERSION, "1")
        for src, expected in EXAMPLES.items():
            with self.subTest(src=src):
                self.assertEqual(" ".join(starmath.atoms(src)), expected)

    def test_same_atoms_as_the_mathml_of_the_generator(self):
        # LibreOffice writes `x^2 over y` as <mfrac><msup>x 2</msup> y</mfrac>
        mathml = parse_xml(b'<math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><msup><mi>x</mi><mn>2</mn></msup>'
                           b'<mi>y</mi></mfrac></math>')
        self.assertEqual(unicode_atoms(mathml_linear(mathml)), starmath.atoms("{x^2} over y"))

    def test_a_wrong_formula_stays_wrong(self):
        self.assertLess(formula_similarity(starmath.atoms("a + b = c"), starmath.atoms("x times y")), FORMULA_RATIO)


class FlatOdtTests(unittest.TestCase):
    FODT = """<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:math="http://www.w3.org/1998/Math/MathML"><office:body><office:text>
<text:table-of-content><text:index-body><text:p>Contents {m1}</text:p></text:index-body></text:table-of-content>
<text:p>The value {m1} grows.</text:p><text:p>{m2}</text:p>
<text:p>Without source {m3}</text:p></office:text></office:body></office:document>"""

    @staticmethod
    def math(src):
        ann = f'<math:annotation encoding="StarMath 5.0">{src}</math:annotation>' if src else ""
        return f"<draw:frame><draw:object><math:math><math:semantics><math:mi>q</math:mi>{ann}</math:semantics></math:math></draw:object></draw:frame>"

    def test_formulas_in_their_paragraphs_indexes_left_out(self):
        root = parse_xml(self.FODT.format(m1=self.math("a^2"), m2=self.math("sqrt x"), m3=self.math("")).encode())
        doc, inv = starmath.fodt_document(root)
        self.assertEqual(inv, {"formula": ["a ^ 2", "√ x"]})
        self.assertEqual(doc.formulas, [(False, "a ^ 2"), (True, "√ x")])
        self.assertEqual([(b.text, b.math) for b in doc.blocks],
                         [("The value grows.", ["a ^ 2"]), ("", ["√ x"]), ("Without source", [])])


if __name__ == "__main__":
    unittest.main()
