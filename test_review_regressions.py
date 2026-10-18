"""Adversarial examples found while reviewing the proposed 0.6 changes."""
from fractions import Fraction
import unittest

from asymptotic_order import compare_bounds, growth
from render_results import render_markdown
from strength_signals import analyze_pairs
from test_strength_signals import pair, side


def analyze(old, new):
    return analyze_pairs([pair(side('paper', old, version=1), side('paper', new, version=2))])[0]


class PremiseReview(unittest.TestCase):
    def properties(self, old, new):
        return analyze(old, new)['algorithm']['new_premises']

    def test_predicate_to_modifier_is_not_new(self):
        for prop, noun, obj in [('planar', 'graph', 'G'), ('compact', 'space', 'X')]:
            old = f'Let ${obj}$ be a {noun}. If ${obj}$ is {prop}, then $P({obj})$ holds.'
            new = f'Let ${obj}$ be a {prop} {noun}. Then $P({obj})$ holds.'
            with self.subTest(prop=prop):
                self.assertEqual(self.properties(old, new), [])
                self.assertEqual(self.properties(new, old), [])

    def test_property_of_another_object_does_not_cover_new_condition(self):
        old = 'Let $G$ be a planar graph. Let $H$ be a graph. Then $H$ is colourable.'
        new = old.replace('Then', 'If $H$ is planar, then')
        findings = self.properties(old, new)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['object_key'], 'H')

    def test_case_sensitive_objects(self):
        old = 'Let $g$ be a graph. Let $G$ be a graph. If $G$ is planar, then $P$ holds.'
        new = old.replace('If $G$ is planar', 'If $g$ is planar')
        findings = self.properties(old, new)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['object_key'], 'g')

    def test_negation_does_not_cover_positive_predicate(self):
        old = 'Let $G$ be a graph. If $G$ is not planar, then $P$ holds.'
        new = old.replace('not planar', 'planar')
        self.assertTrue(self.properties(old, new))

    def test_modifier_replacement_is_not_generic_addition(self):
        old = 'Let $G$ be a finite graph. Then $P(G)$ holds.'
        new = old.replace('finite', 'infinite')
        self.assertFalse(any(p['property'].startswith('modifier:') for p in self.properties(old, new)))

    def test_rebound_object_does_not_match_first_declaration(self):
        old = 'Let $G$ be a graph. Let $G$ be a planar graph. Then $P(G)$ holds.'
        new = old.replace('be a graph', 'be a connected graph', 1)
        self.assertFalse(any(p['property'].startswith('modifier:') for p in self.properties(old, new)))

    def test_generic_rule_does_not_join_enumeration_items(self):
        old = r'\begin{enumerate}\item Let $G$ be a graph. Then $P$ holds.\item Let $G$ be a planar graph. Then $Q$ holds.\end{enumerate}'
        new = old.replace('be a graph', 'be a planar graph', 1)
        self.assertEqual(self.properties(old, new), [])


class AsymptoticReview(unittest.TestCase):
    def compare(self, old, new, limit=r'n\to\infty'):
        prefix = 'As $' + limit + '$. Then '
        return compare_bounds(prefix + old, prefix + new)

    def test_zero_limit_reverses_order(self):
        findings = self.compare('$f(x)=O(x)$.', '$f(x)=O(x^2)$.', r'x\to0')
        self.assertEqual(findings[0]['direction'], 'stronger')
        self.assertEqual(findings[0]['limit'], 'zero')

    def test_unknown_or_conflicting_limits_abstain(self):
        old, new = 'Then $f(n)=O(n)$.', 'Then $f(n)=O(n^2)$.'
        self.assertEqual(compare_bounds(old, new), [])
        self.assertEqual(compare_bounds(r'As $n\to0$. ' + old, r'As $n\to\infty$. ' + new), [])
        for limit in [r'n\to1', r'n\to-\infty', r'n\to0$ and $n\to\infty']:
            self.assertEqual(self.compare('$f(n)=O(n)$.', '$f(n)=O(n^2)$.', limit), [])

    def test_commented_limit_does_not_enable_comparison(self):
        prefix = '% As $n\\to\\infty$\nThen '
        self.assertEqual(compare_bounds(prefix + '$f(n)=O(n)$.', prefix + '$f(n)=O(n^2)$.'), [])

    def test_unknown_control_word_is_not_split_into_known_function(self):
        self.assertIsNone(growth(r'\logn'))
        self.assertEqual(growth(r'\log n'), ('n', Fraction(0), Fraction(1)))

    def test_nested_condition_is_not_conclusion_bound(self):
        self.assertEqual(self.compare('if $f(n)=O(n)$, the series converges.',
                                      'if $f(n)=O(n^2)$, the series converges.'), [])
        self.assertEqual(self.compare('for every function such that $f(n)=O(n)$, the series converges.',
                                      'for every function such that $f(n)=O(n^2)$, the series converges.'), [])

    def test_prose_or_empty_subject_is_not_paired(self):
        self.assertEqual(self.compare('The cost is $O(n)$.', 'The error is $O(n^2)$.'), [])

    def test_inequalities_and_composite_rhs_abstain(self):
        for pattern in [r'$f(n)\geq O(%s)$.', r'$f(n)=O(%s)+n^2$.', r'$f(n)=O(%s)=g(n)$.']:
            self.assertEqual(self.compare(pattern % 'n', pattern % 'n^2'), [])

    def test_operator_changes_not_erased(self):
        self.assertEqual(self.compare('$f(n)=o(n)$.', '$f(n)=O(n^2)$.'), [])
        self.assertEqual(self.compare('$f(n)=O(n)$.', '$f(n)=o(n)$.'), [])

    def test_unbraced_exponent_and_division(self):
        self.assertEqual(growth('n^2/3'), ('n', Fraction(2), Fraction(0)))
        self.assertEqual(growth('n^12'), ('n', Fraction(1), Fraction(0)))
        self.assertEqual(growth('n^{12}'), ('n', Fraction(12), Fraction(0)))
        self.assertEqual(self.compare('$f(n)=O(n^2/3)$.', '$f(n)=O(n)$.')[0]['direction'], 'stronger')

    def test_decimal_fraction_and_zero_denominator(self):
        self.assertEqual(growth('n^{1.5/2}'), ('n', Fraction(3, 4), Fraction(0)))
        for expression in [r'n^{1/0}', r'n^{\frac{1}{0}}', '0n', 'n/0', 'n^{', '/2n']:
            self.assertIsNone(growth(expression), expression)

    def test_log_near_zero_abstains(self):
        self.assertEqual(self.compare(r'$f(x)=O(x\log x)$.', '$f(x)=O(x^2)$.', r'x\to0'), [])

    def test_unsupported_expression_does_not_abort_batch(self):
        prefix = r'As $n\to\infty$. Then '
        pairs = [pair(side('p', prefix + '$T(n)=O(n)$.', version=1),
                      side('p', prefix + '$T(n)=O(' + expression + ')$.', version=2), pair_id=str(i))
                 for i, expression in enumerate(['n^{1.5/2}', 'n^{1/0}', 'n^2'])]
        results = analyze_pairs(pairs)
        self.assertEqual(len(results), 3)
        self.assertTrue(any(s['kind'] == 'asymptotic_bound_change' for s in results[-1]['algorithm']['signals']))

    def test_unsupported_change_cannot_gain_masked_alignment(self):
        result = analyze('Then $f(n)=O(n)$.', 'Then $f(n)=O(e^n)$.')
        self.assertFalse(any(s['kind'] == 'asymptotic_bound_change' for s in result['algorithm']['signals']))


class ReportReview(unittest.TestCase):
    def test_named_theorem_keeps_change_in_heading(self):
        old = r'\begin{theorem}[Colouring] Let $G$ be a graph. Then $G$ is $5$-colourable.\end{theorem}'
        result = analyze(old, old.replace('a graph', 'a planar graph'))
        heading = next(line for line in render_markdown([result]).splitlines() if line.startswith('## 1.'))
        self.assertIn('planar', heading)
        self.assertIn('Colouring', heading)


if __name__ == '__main__':
    unittest.main()
