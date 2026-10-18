"""Adversarial tests of actual local condition direction and scope separation."""
import unittest
from formula_scope import analyze_formula_scope, added_formula_premises
from strength_signals import _math_units, _formula_evidence, analyze_pairs
from test_strength_signals import pair, side

def analyze(a,b):
    return analyze_formula_scope(a,b,_math_units,_formula_evidence)

def findings(a,b):
    return analyze(a,b)['findings']

class TestFormulaScope(unittest.TestCase):
    def kinds(self,a,b):
        return {f['kind'] for f in findings(a,b)}

    def test_lower_bound_tightens(self):
        fs=findings(r'If $n\geq0$, then $P(n)$.',r'If $n\geq2$, then $P(n)$.')
        self.assertEqual(fs[0]['kind'],'tightened')
        self.assertTrue(fs[0]['eligible_as_added_premise'])

    def test_reverse_direction_relaxes(self):
        self.assertIn('relaxed',self.kinds(r'If $n>2$, then P.',r'If $n>0$, then P.'))

    def test_strict_endpoint(self):
        self.assertIn('tightened',self.kinds(r'Let $p\leq1$. Then P.',r'Let $p<1$. Then P.'))

    def test_chain_keeps_both_ends(self):
        self.assertIn('tightened',self.kinds(r'If $0<p<2$, then P.',r'If $0<p<1$, then P.'))

    def test_crossing_intervals_direction_unknown(self):
        self.assertIn('replaced',self.kinds(r'If $0<p<2$, then P.',r'If $1<p<3$, then P.'))

    def test_closed_to_open_membership(self):
        self.assertIn('tightened',self.kinds(r'Let $p\in[0,1]$. Then P.',r'Let $p\in[0,1)$. Then P.'))

    def test_equal_symbolic_endpoint(self):
        self.assertIn('tightened',self.kinds(r'Let $p\in[0,a]$. Then P.',r'Let $p\in[0,a)$. Then P.'))

    def test_unequal_symbolic_endpoint_not_ordered(self):
        self.assertIn('replaced',self.kinds(r'Let $p\in[0,a]$. Then P.',r'Let $p\in[0,b]$. Then P.'))

    def test_standard_set_containment(self):
        fs=findings(r'For every $x\in\mathbb{R}$, P.',r'For every $x\in\mathbb{Q}$, P.')
        self.assertEqual(fs[0]['kind'],'tightened')
        self.assertTrue(fs[0]['eligible_as_added_premise'])

    def test_natural_number_zero_convention_not_inferred(self):
        fs=findings(r'Let $n\in\mathbb{Z}$. Then P.',r'Let $n\in\mathbb{N}$. Then P.')
        self.assertIn('是否含零',fs[0]['comparison_rule'])

    def test_custom_set_macros_not_assumed_standard(self):
        self.assertIn('replaced',self.kinds(r'Let $x\in\R$. Then P.',r'Let $x\in\Q$. Then P.'))

    def test_added_nonzero_formula(self):
        fs=findings(r'Let $x\in X$. Then P.',r'Let $x\in X$ with $x\neq0$. Then P.')
        self.assertTrue(any(f['kind']=='added_condition' and f['eligible_as_added_premise'] for f in fs))

    def test_new_binder_only_domain_is_separate(self):
        fs=findings(r'For every $x$, $P(x)$.',r'Let $x\in X$. For every $x$, $P(x)$.')
        self.assertTrue(any(f['kind']=='newly_explicit_domain' and not f['eligible_as_added_premise'] for f in fs))

    def test_new_unrelated_object_not_aligned(self):
        self.assertNotIn('added_condition',self.kinds(r'Let $x\in X$. Then P.',r'Let $y\in Y$ with $y>0$. Then P.'))

    def test_case_sensitive_objects(self):
        self.assertNotIn('added_condition',self.kinds(r'Let $X\in V$. Then P.',r'Let $x\in V$ with $x>0$. Then P.'))

    def test_macro_substring_not_object(self):
        self.assertNotIn('added_condition',self.kinds(r'Let $\exp(t)$ be given.',r'Let $x>0$. Then P.'))

    def test_conclusion_bound_not_premise(self):
        fs=findings(r'Let $n\geq0$. Then $f(n)\leq2$.',r'Let $n\geq0$. Then $f(n)\leq1$.')
        self.assertTrue(any(f['kind']=='tightened' for f in fs))
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_existential_bound_not_premise(self):
        fs=findings(r'There exists $C>0$ such that P.',r'There exists $C>1$ such that P.')
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_biconditional_not_promoted(self):
        fs=findings(r'P holds if and only if $x>0$.',r'P holds if and only if $x>1$.')
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_explicit_implication(self):
        fs=findings(r'$n\geq0\Rightarrow P(n)$.',r'$n\geq2\Rightarrow P(n)$.')
        self.assertTrue(any(f['eligible_as_added_premise'] for f in fs))

    def test_or_condition_not_conjunction(self):
        fs=findings(r'If $x>0$ or $x<-2$, then P.',r'If $x>1$ or $x<-2$, then P.')
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_definition_not_restricted_premise(self):
        fs=findings(r'Let $c:=1$. Then P.',r'Let $c:=2$. Then P.')
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_numeric_index_inside_sum_not_global_constraint(self):
        fs=findings(r'If $\sum_{n=1}^N a_n>0$, then P.',r'If $\sum_{n=2}^N a_n>0$, then P.')
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_set_builder_not_global_domain(self):
        fs=findings(r'Let $S=\{x\in X\mid x>0\}$. Then P.',r'Let $S=\{x\in X\mid x>1\}$. Then P.')
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_decimal_numeric_not_lexical_order(self):
        self.assertIn('tightened',self.kinds(r'If $n>2$, then P.',r'If $n>10$, then P.'))

    def test_rational_bound_exact(self):
        self.assertIn('tightened',self.kinds(r'If $p>\frac{1}{3}$, then P.',r'If $p>\frac{1}{2}$, then P.'))

    def test_ascii_bound(self):
        self.assertIn('tightened',self.kinds('If $n>=0$, then P.','If $n>=1$, then P.'))

    def test_multiple_members_separated(self):
        fs=findings(r'Let $a,b\in X$. Then P.',r'Let $a,b\in Y$. Then P.')
        self.assertEqual({f['object_key'] for f in fs},{'a','b'})

    def test_relation_unchanged_after_moving_is_not_addition(self):
        fs=findings(r'If $x>0$ and $y>0$, then P.',r'If $y>0$ and $x>0$, then P.')
        self.assertFalse(fs)

    def test_removed_condition_not_new_premise(self):
        fs=findings(r'If $x>0$ and $y>0$, then P.',r'If $x>0$, then $P(y)$.')
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))
        self.assertIn('removed_condition',{f['kind'] for f in fs})

    def test_symbolic_relation_change_retained_without_direction(self):
        self.assertIn('replaced',self.kinds(r'If $x<y$, then P.',r'If $x<z$, then P.'))

    def test_new_premise_coordinates_are_exact(self):
        a=r'Let $x\in X$. Then P.';b=r'Let $x\in X$ with $x\neq0$. Then P.'
        ps=added_formula_premises(analyze(a,b),a,b)
        self.assertTrue(ps)
        for p in ps:
            self.assertEqual(p['new_minimal'],b[slice(*p['new_span'])])

    def test_engine_exposes_new_premise(self):
        p=pair(side('paper',r'If $n\geq0$, then $P(n)$ holds for all x.',version=1),
               side('paper',r'If $n\geq2$, then $P(n)$ holds for all x.',version=2))
        a=analyze_pairs([p])[0]['algorithm']
        self.assertTrue(a['new_premises'])
        self.assertEqual(a['change_analysis']['category'],'premise_change')

    def test_postposed_universal_nonzero_ideal(self):
        a=r'Then $F(a)\leq G(a)$ for every ideal sheaf $a$ on X.'
        b=r'Then $F(a)\leq G(a)$ for every nonzero ideal sheaf $a$ on X.'
        result=analyze_pairs([pair(side('p',a,version=1),side('p',b,version=2))])[0]['algorithm']
        self.assertTrue(any(p['property']=='nonzero' for p in result['new_premises']))

    def test_irreducible_representation_property(self):
        a=r'Let $V$ be a finite-dimensional representation of G. Then F is valid.'
        b=r'Let $V$ be a finite-dimensional irreducible representation of G. Then F is valid.'
        result=analyze_pairs([pair(side('p',a,version=1),side('p',b,version=2))])[0]['algorithm']
        self.assertTrue(any(p['property']=='irreducible' for p in result['new_premises']))

    def test_formula_condition_reference_change_not_automatically_added(self):
        fs=findings('Under regularity conditions (1)--(6), P.','Under regularity conditions (1)--(8), P.')
        self.assertIn('reference_scope_changed',{f['kind'] for f in fs})
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_empty_interval_not_declared_weakening(self):
        self.assertIn('replaced',self.kinds(r'Let $x\in[0,1]$. Then P.',r'Let $x\in(1,1)$. Then P.'))

    def test_repeated_constraint_multiplicity_not_silent_addition(self):
        fs=findings(r'If $x>0$, then P.',r'If $x>0$ and $x>0$, then P.')
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_branch_conditions_are_local(self):
        a=r'Let $x\in X$. Then \[f(x)=\begin{cases}1&x>0\\0&x\leq0\end{cases}\]'
        b=r'Let $x\in X$. Then \[f(x)=\begin{cases}1&x>1\\0&x\leq1\end{cases}\]'
        self.assertFalse(any(f['eligible_as_added_premise'] for f in findings(a,b)))

    def test_shared_numeric_condition_applies_to_both_parameters(self):
        fs=findings(r'Let $\beta,\epsilon>0$. Then P.',r'Let $\epsilon>0$ and $\beta\geq0$. Then P.')
        self.assertTrue(any(f['kind']=='relaxed' and f['object_key']==r'\beta' for f in fs))
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_compound_left_side_changed_not_new_condition(self):
        fs=findings(r'Let $a$ satisfy $f(a)+1=b$. Then P.',r'Let $a$ satisfy $f(a)=b+1$. Then P.')
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_conclusion_after_forall_not_a_parameter_condition(self):
        fs=findings(r'For every $n>0$, the following estimate holds: $f(n)<2$.',r'For every $n>0$, the following estimate holds: $f(n)<1$.')
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_flat_interval_rewrite_filters_complete_change(self):
        a=side('p',r'Let $r\geq4$. Then P holds.',version=1)
        b=side('p',r'Let $r\in[4,\infty)$. Then P holds.',version=2)
        self.assertTrue(analyze_pairs([pair(a,b)])[0]['algorithm']['rewrite_equivalence']['suppresses_strength_alert'])

    def test_interval_rewrite_does_not_hide_conclusion_change(self):
        a=side('p',r'Let $r\geq4$. Then $P(r)=1$.',version=1)
        b=side('p',r'Let $r\in[4,\infty)$. Then $P(r)=2$.',version=2)
        self.assertFalse(analyze_pairs([pair(a,b)])[0]['algorithm']['rewrite_equivalence']['suppresses_strength_alert'])

    def test_color_prefix_comment_cannot_change_role(self):
        a=side('p','Then $x>0$. % assume\n',version=1)
        b=side('p','Then $x>1$. % assume\n',version=2)
        self.assertFalse(analyze_pairs([pair(a,b)])[0]['algorithm']['new_premises'])

    def test_same_symbol_in_separate_items_not_intersected(self):
        a=r'\begin{enumerate}\item Let $r>0$. Then A. \item Let $r\geq0$. Then B.\end{enumerate}'
        b=r'\begin{enumerate}\item Let $r>0$. Then A. \item Let $r>0$. Then B.\end{enumerate}'
        self.assertTrue(any(f['kind']=='tightened' and f['object_key']=='r' for f in findings(a,b)))

    def test_ellipsis_not_false_point_assumption(self):
        fs=findings(r'For any $k=0,\dots,N$, P.',r'For any $k\neq0$, P.')
        self.assertFalse(any(f.get('old_atoms')==['k=0'] for f in fs))

    def test_changed_compound_condition_exposed_not_certified(self):
        fs=findings(r'Suppose $t+c s<0$. Then P.',r'Suppose $t+s<0$. Then P.')
        self.assertIn('condition_expression_changed',{f['kind'] for f in fs})
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_case_only_quantifier_change_not_scope_event(self):
        self.assertNotIn('quantifier_scope_changed',self.kinds('For every $x$, P.','for every $x$, P.'))

    def test_standard_product_set_scope(self):
        fs=findings(r'$S\subset\Lambda\times\mathbb Z^N$ is a monoid.',r'$S\subset\Lambda\times\mathbb N^N$ is a monoid.')
        self.assertTrue(any(f['kind']=='tightened' for f in fs))

    def test_standard_set_typography_does_not_relax(self):
        fs=findings(r'Let $x\in\mathbb{Q}$. Then P.',r'Let $x\in\mathbb Q$. Then P.')
        self.assertFalse(any(f.get('local_condition_direction') in {'tightened','relaxed'} for f in fs))

    def test_explicit_nonzero_set_exclusion(self):
        fs=findings(r'Let $x\in X$. Then P.',r'Let $x\in X\setminus\{0\}$. Then P.')
        self.assertTrue(any(f['kind']=='tightened' and f['eligible_as_added_premise'] for f in fs))

    def test_negated_prose_condition_not_interval_assumption(self):
        fs=findings(r'If not $x>0$, then P.',r'If not $x>1$, then P.')
        self.assertFalse(any(f['eligible_as_added_premise'] for f in fs))

    def test_new_bound_keeps_domain_global_assumptions_unknown(self):
        fs=findings(r'If $n\geq0$, then P.',r'If $n\geq1$, then P.')
        self.assertTrue(fs)
        self.assertFalse(fs[0]['global_weakening_certified'])
        self.assertFalse(fs[0]['global_assumptions_checked'])

    def test_complex_manifold_smoothness_is_not_added_premise(self):
        a=side('p','Let $X$ be a complex manifold. Then P holds.',version=1)
        b=side('p','Let $X$ be a smooth complex manifold. Then P holds.',version=2)
        result=analyze_pairs([pair(a,b)])[0]['algorithm']
        self.assertFalse(result['new_premises'])
        self.assertTrue(result['rewrite_equivalence']['suppresses_strength_alert'])

    def test_smooth_analytic_space_is_not_redundant(self):
        a=side('p','Let $X$ be a complex analytic space. Then P holds.',version=1)
        b=side('p','Let $X$ be a smooth complex analytic space. Then P holds.',version=2)
        self.assertFalse(analyze_pairs([pair(a,b)])[0]['algorithm']['rewrite_equivalence']['suppresses_strength_alert'])

    def test_redundant_type_word_does_not_hide_changed_bound(self):
        a=side('p','Let $X$ be a complex manifold. Then $f(X)<2$.',version=1)
        b=side('p','Let $X$ be a smooth complex manifold. Then $f(X)<1$.',version=2)
        self.assertFalse(analyze_pairs([pair(a,b)])[0]['algorithm']['rewrite_equivalence']['suppresses_strength_alert'])

    def test_transverse_adverb_not_a_strength_change(self):
        a=side('p','Let $F$ be transversally smooth. Then P holds.',version=1)
        b=side('p','Let $F$ be transversely smooth. Then P holds.',version=2)
        self.assertTrue(analyze_pairs([pair(a,b)])[0]['algorithm']['rewrite_equivalence']['suppresses_strength_alert'])

    def test_added_condition_anchor_is_in_same_item_math_not_an_english_letter(self):
        old=r'The following conditions hold. \begin{enumerate}\item $P(n)$.\item $Q(n)$.\end{enumerate}'
        new=r'The following conditions hold. \begin{enumerate}\item $P(n)$.\item $Q(n)$ for every $n>2$.\end{enumerate}'
        analysis=analyze(old,new)
        ps=added_formula_premises(analysis,old,new,_math_units)
        self.assertTrue(ps)
        self.assertEqual(ps[0]['old_minimal'],'$Q(n)$')
        self.assertEqual(ps[0]['old_anchor_kind'],'same_symbol_formula_context_not_an_old_condition')

    def test_estimate_condition_does_not_restrict_earlier_existence(self):
        old=r'For every $k$, there exists $u_k$ solving \[L u_k=f_k.\] such that for every $a>0$ \[\|u_k\|\leq C.\]'
        new=r'For every $k$, there exists $u_k$ solving \[L u_k=f_k.\] such that for every $a>0$, $k\neq0$, \[\|u_k\|\leq C.\label{estimate}\]'
        fs=findings(old,new)
        k=next(f for f in fs if f.get('object_key')=='k' and f['kind']=='added_condition')
        scope=k['new_guarantee_scopes'][0]
        self.assertEqual(scope['extent'],'local_following_formula')
        self.assertEqual(scope['target']['tex_labels'],['estimate'])
        self.assertIn('L u_k',scope['earlier_formula_in_same_item']['formula_quote'])
        self.assertFalse(scope['whole_statement_scope_certified'])

    def test_initial_parameter_condition_not_mislabelled_post_estimate(self):
        fs=findings(r'Let $k\geq0$. Then \[f(k)=1.\]',r'Let $k>0$. Then \[f(k)=1.\]')
        self.assertEqual(fs[0]['new_guarantee_scopes'][0]['extent'],'same_item_following_formula')

    def test_condition_reference_numbers_exclude_initial_value_math(self):
        a=r'Under regularity conditions (7)--(9), with initial conditions $m(0)=f(0)$, P.'
        b=r'Under regularity conditions (9), (11), (12), with initial conditions $m(0)=f(0)$, P.'
        f=next(f for f in findings(a,b) if f['kind']=='reference_scope_changed')
        self.assertEqual(f['old_reference_numbers'],[['7','9']])
        self.assertEqual(f['new_reference_numbers'],[['9','11','12']])

    def test_function_argument_is_not_a_condition_list_number(self):
        a=r'Assume initial conditions $m(0)=f(0)$. Then P.'
        b=r'Assume initial conditions $m(1)=f(1)$. Then P.'
        self.assertNotIn('reference_scope_changed',{f['kind'] for f in findings(a,b)})

if __name__=='__main__':
    unittest.main()
