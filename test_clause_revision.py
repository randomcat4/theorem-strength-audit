import copy
import unittest
from premise_clauses import clauses, comparison, reference_evidence
from strength_signals import analyze_pairs
from render_results import render_markdown, summarize_result


def pair(a, b, key='synthetic'):
    return {'pair_id': key, 'old': {'paper_id': 'p', 'version': 1, 'text': a},
            'new': {'paper_id': 'p', 'version': 2, 'text': b}}


class ClauseBoundaries(unittest.TestCase):
    def test_formula_commas_and_multiline_are_inside_clause(self):
        t = 'Let $z$ be a point. If $z$ lies in $U(a,b)$\n and $V(c,d)$, then $P(z)$ holds.'
        c = clauses(t)[0]
        self.assertEqual(c.group(), 'If $z$ lies in $U(a,b)$\n and $V(c,d)$')
        self.assertEqual(t[c.start():c.end()], c.group())

    def test_footnote_predicate_and_period_cannot_end_main_clause(self):
        t = r'Let $X$ be a space. If $X$\footnote{If $Y$ is compact, then $Q$. Nested {a,b}.} is locally compact, then $P$.'
        c = clauses(t)
        self.assertEqual(len(c), 1)
        self.assertTrue(c[0].group().endswith('is locally compact'))
        self.assertEqual(c[0].group(), t[c[0].start():c[0].end()])

    def test_parenthetical_abbreviation_and_braces(self):
        t = r'Assume $X$ is (a.s.) regular and \textbf{locally finite, nonempty}, then $P$.'
        self.assertTrue(clauses(t)[0].group().endswith('nonempty}'))

    def test_incomplete_or_overbound_never_promoted(self):
        for t in ['If $X is planar, then $P$.', 'If $X$ is '+('a '*4000)+', then P.', r'If $X$\footnote{missing is compact, then P.']:
            self.assertEqual(clauses(t), [])

    def test_note_property_does_not_become_declaration_property(self):
        a = 'Let $X$ be a space. Then $P(X)$.'
        b = r'Let $X$ be a space\footnote{A compact space has another property.}. Then $P(X)$.'
        result = analyze_pairs([pair(a,b)])[0]['algorithm']
        self.assertEqual(result['new_premises'], [])
        self.assertTrue(result['footnote_changes'])

    def test_changed_predicate_keeps_old_complete_clause_not_object(self):
        a = 'Let $z$ be a point. If $z$ is admissible, then $P(z)$.'
        b = a.replace('is admissible', 'is admissible for $F(u,v)$')
        r = analyze_pairs([pair(a,b)])[0]['algorithm']
        self.assertEqual(r['new_premises'], [])
        self.assertEqual(r['hypothesis_comparison'][0]['old_quote'], 'If $z$ is admissible')
        self.assertIn('$F(u,v)$', r['hypothesis_comparison'][0]['new_quote'])
        self.assertFalse(r['rewrite_equivalence']['suppresses_strength_alert'])

    def test_truly_added_independent_condition_keeps_old_declaration(self):
        a = 'Let $H$ be a graph. Then $P(H)$.'
        b = a.replace('Then', 'If $H$ is bipartite, then')
        r = analyze_pairs([pair(a,b)])[0]['algorithm']
        self.assertEqual(r['new_premises'][0]['old_minimal'], 'Let $H$ be a graph')
        self.assertEqual(r['new_premises'][0]['new_minimal'], 'If $H$ is bipartite')

    def test_changed_existing_conjunction_is_not_lost(self):
        a = 'Let $H$ be a graph. If $H$ is bipartite, then $P(H)$.'
        b = a.replace('is bipartite', 'is bipartite and connected')
        r = analyze_pairs([pair(a,b)])[0]['algorithm']
        self.assertEqual(r['hypothesis_comparison'][0]['status'], 'changed_existing_clause')
        self.assertIn('connected', r['hypothesis_comparison'][0]['new_quote'])


class ReferenceExpansion(unittest.TestCase):
    def fixture(self):
        a = r'Assume the following operators are bounded: \begin{equation}\label{ops}A(x,y),B(z).\end{equation} Then $P$.'
        b = r'Assume the operators in \eqref{ops} are bounded. Then $P$.'
        p = pair(a,b)
        p['evidence_documents'] = [
            {'source_type':'full_tex','side':'old','paper_id':'p','version':1,'text':a},
            {'source_type':'full_tex','side':'new','paper_id':'p','version':2,'text':r'\begin{equation}\label{ops}A(x,y),B(z).\end{equation}'+b}]
        return p

    def evidence(self,p): return reference_evidence(p,p['old'],p['new'],p['old']['text'],p['new']['text'])

    def test_moved_list_compares_actual_targets(self):
        p=self.fixture()
        e=self.evidence(p)
        self.assertEqual(e['comparisons'][0]['status'],'same_literal_equation')
        self.assertIn('A(x,y)',e['comparisons'][0]['new']['quote'])
        self.assertEqual(clauses(p['old']['text'])[0].group().count('Assume'),1)

    def test_same_label_changed_content_not_suppressed(self):
        p=self.fixture();p['evidence_documents'][1]['text']=p['evidence_documents'][1]['text'].replace('B(z)','C(z)')
        self.assertEqual(self.evidence(p)['comparisons'][0]['status'],'changed_equation')

    def test_missing_wrong_version_duplicate_and_truncated_stay_unknown(self):
        for case in ['missing','version','duplicate','truncated']:
            p=self.fixture();d=p['evidence_documents'][1]
            if case=='missing': d['text']=p['new']['text']
            if case=='version': d['version']=3
            if case=='duplicate': d['text']+=r'\begin{equation}\label{ops}C.\end{equation}'
            if case=='truncated': d['text_truncated']=True
            self.assertEqual(self.evidence(p)['endpoints']['new']['ops']['status'],'unresolved_or_ambiguous')

    def test_comments_do_not_supply_target(self):
        p=self.fixture();p['evidence_documents'][1]['text']='% '+p['evidence_documents'][1]['text'].split('Assume')[0]+'\n'+p['new']['text']
        self.assertEqual(self.evidence(p)['comparisons'],[])

    def test_multiple_sources_and_inactive_targets_are_not_resolved(self):
        p=self.fixture()
        p['evidence_documents'].append({**p['evidence_documents'][1], 'text':p['evidence_documents'][1]['text']+'\nOther body.'})
        self.assertEqual(self.evidence(p)['comparisons'],[])
        for prefix in [r'\iffalse ',r'\end{document}']:
            p=self.fixture();p['evidence_documents'][1]['text']=prefix+p['evidence_documents'][1]['text']
            self.assertEqual(self.evidence(p)['comparisons'],[])


class RepeatedReading(unittest.TestCase):
    def test_restatement_group_preserves_rows_and_merges_reading(self):
        a=r'Let $x$ be real. Then \begin{equation}f(x)>0.\end{equation}'
        b=a.replace('real','positive')
        p=pair(a,b,'first');q=copy.deepcopy(p);q['pair_id']='second'
        q['new']['text']=b.replace('{equation}','{equation*}')
        results=analyze_pairs([p,q]);self.assertEqual(len(results),2)
        self.assertEqual(results[0]['algorithm']['reading_group']['pair_count'],2)
        md=render_markdown(results)
        self.assertIn('1 个阅读组',md);self.assertEqual(md.count('## 1.'),1);self.assertNotIn('## 2.',md)
        self.assertIn('同组其他出处',md)
        self.assertTrue(summarize_result(results[1])['algorithm']['reading_group']['repeated_text'])

    def test_different_version_or_formula_is_not_merged(self):
        p=pair('Let $x$ be real. Then $P(x)$.','Let $x$ be positive. Then $P(x)$.','first')
        for variant in ['version','formula','case']:
            q=copy.deepcopy(p);q['pair_id']='second'
            if variant=='version':q['new']['version']=3
            elif variant=='formula':q['new']['text']=q['new']['text'].replace('P(x)','Q(x)')
            else:q['new']['text']=q['new']['text'].replace('$x$','$X$')
            results=analyze_pairs([p,q])
            self.assertFalse(results[0]['algorithm']['reading_group']['repeated_text'])

    def test_prose_word_boundaries_are_not_erased(self):
        from revision_groups import literal_key
        self.assertNotEqual(literal_key('a part'),literal_key('apart'))

    def test_ambiguous_old_predicates_cannot_be_claimed_absent(self):
        a='Let $G$ be a graph. Assume $G$ is finite. Assume $G$ is planar. Then $P(G)$.'
        b='Let $G$ be a graph. Assume $G$ is connected. Then $P(G)$.'
        r=analyze_pairs([pair(a,b)])[0]['algorithm']
        self.assertFalse(any(p['property'].startswith('hypothesis:') for p in r['new_premises']))

    def test_metadata_precedes_diagnostics_and_full_clause_visible(self):
        a='Let $z$ be a point. If $z$ is admissible, then $P(z)$.'
        b=a.replace('admissible','admissible for $F(u,v)$')
        md=render_markdown(analyze_pairs([pair(a,b)]))
        card=md.split('## 1.',1)[1]
        self.assertLess(card.index('| 论文作者姓名 |'),card.index('**首要缺口：**'))
        visible=card.split('<details>',1)[0]
        self.assertIn(r'If \$z\$ is admissible',visible)
        self.assertIn(r'\$F(u,v)\$',visible)


if __name__=='__main__':unittest.main()
