import unittest

from asymptotic_order import compare_bounds, growth
from strength_signals import analyze_pairs
from test_strength_signals import pair, side


class Growth(unittest.TestCase):
    def test_supported_products(self):
        self.assertEqual(growth(r"n\log n")[1:], (1, 1))
        self.assertEqual(growth(r"n^{3/2}")[1:], (1.5, 0))
        self.assertEqual(growth(r"\sqrt{n}\log^2 n")[1:], (0.5, 2))
        self.assertEqual(growth(r"(\log n)^{2}")[1:], (0, 2))
        self.assertEqual(growth(r"n^{\frac{1}{2}}")[1:], (0.5, 0))
        self.assertEqual(growth("2n")[1:], (1, 0))

    def test_unsupported_expressions_are_not_compared(self):
        for text in ("n+1", "e^n", r"n\log m", r"\frac{n}{\log n}", r"\log\log n"):
            with self.subTest(text):
                self.assertIsNone(growth(text))


class CompareBounds(unittest.TestCase):
    def directions(self, old, new):
        return [(f["kind"], f["direction"]) for f in compare_bounds(old, new)]

    def test_upper_bound_relaxed_is_weaker(self):
        self.assertEqual(self.directions(r"Then $T(n)=O(n\log n)$.", r"Then $T(n)=O(n^{2})$."), [("upper", "weaker")])

    def test_upper_bound_tightened_is_stronger(self):
        self.assertEqual(self.directions(r"Then $e(G)=O(n^{3/2})$.", r"Then $e(G)=O(n^{4/3})$."), [("upper", "stronger")])

    def test_lower_bound_lowered_is_weaker(self):
        self.assertEqual(self.directions(r"Then $e(G)=\Omega(n^2)$.", r"Then $e(G)=\Omega(n\log n)$."), [("lower", "weaker")])

    def test_theta_change_has_no_direction(self):
        self.assertEqual(self.directions(r"Then $T(n)=\Theta(n)$.", r"Then $T(n)=\Theta(n\log n)$."), [("tight", "changed_tight_bound")])

    def test_hypothesis_bounds_and_different_subjects_are_not_paired(self):
        self.assertEqual(self.directions(r"Let $f(n)=O(n)$. Then $g(n)=O(n)$.", r"Let $f(n)=O(n^2)$. Then $g(n)=O(n)$."), [])
        self.assertEqual(self.directions(r"Then $f(n)=O(n)$.", r"Then $g(n)=O(n^2)$."), [])
        self.assertEqual(self.directions(r"Suppose $f(n)=O(n)$.", r"Suppose $f(n)=O(n^2)$."), [])

    def test_equal_orders_are_not_reported(self):
        self.assertEqual(self.directions(r"Then $T(n)=O(2n)$.", r"Then $T(n)=O(n)$."), [])


class Integration(unittest.TestCase):
    def analyze(self, old, new):
        return analyze_pairs([pair(side("paper", old, version=1), side("paper", new, version=2))])[0]["algorithm"]

    def test_relaxed_bound_on_short_statement_is_aligned_and_ranked(self):
        algo = self.analyze(r"Let $G$ be a graph on $n$ vertices. Then $e(G)=\Omega(n^{2})$.",
                            r"Let $G$ be a graph on $n$ vertices. Then $e(G)=\Omega(n\log n)$.")
        self.assertEqual(algo["alignment_evidence"]["status"], "strong_textual_overlap")
        self.assertIn("渐近界放宽", algo["headline"])
        self.assertIn("asymptotic_bound_change", {s["kind"] for s in algo["signals"]})

    def test_tightened_bound_scores_below_relaxed_bound(self):
        old = r"Let $G$ be a graph on $n$ vertices. Then $e(G)=O(n^{3/2})$."
        tighter = self.analyze(old, r"Let $G$ be a graph on $n$ vertices. Then $e(G)=O(n^{4/3})$.")
        looser = self.analyze(old, r"Let $G$ be a graph on $n$ vertices. Then $e(G)=O(n^{5/3})$.")
        self.assertLess(tighter["priority_score"], looser["priority_score"])
        self.assertNotIn("渐近界放宽", tighter["headline"])


if __name__ == "__main__":
    unittest.main()
