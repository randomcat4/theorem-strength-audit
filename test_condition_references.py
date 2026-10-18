import copy
import unittest
from condition_references import compare_condition_references, numbered_items
from premise_clauses import comparison, unmatched_clauses
from test_clause_revision import pair


class NumberedReferences(unittest.TestCase):
    def fixture(self):
        a=r'Assume conditions (1)--(2) in Appendix \ref{conditions} hold. Then $P$.'
        b=a.replace('(2)','(3)')
        p=pair(a,b)
        p['evidence_documents']=[
            dict(source_type='full_tex',side='old',paper_id='p',version=1,text=a+r'\section{Conditions}\label{conditions}\begin{enumerate}\item $X$ is compact.\item $X$ is connected.\end{enumerate}\begin{itemize}\item[(3)] $X$ is smooth.\end{itemize}'),
            dict(source_type='full_tex',side='new',paper_id='p',version=2,text=b+r'\section{Conditions}\label{conditions}\begin{itemize}\item[(1)] $X$ is compact.\item[(2)] $X$ is smooth.\item[(3)] $X$ is connected.\end{itemize}')]
        return p

    def run_refs(self,p):return compare_condition_references(p,p['old'],p['new'],p['old']['text'],p['new']['text'])

    def test_renumbering_and_reference_inclusion_are_separate(self):
        r=self.run_refs(self.fixture());c=r['comparisons'][0]
        self.assertEqual(c['mapping'][1]['old_number'],3)
        self.assertEqual(c['mapping'][1]['status'],'newly_referenced_existing_literal_item')
        self.assertEqual(c['mapping'][2]['old_number'],2)
        self.assertEqual(c['mapping'][2]['status'],'same_literal_selected_item')
        self.assertIsNone(c['new_condition_count'])

    def test_changed_content_same_number_is_not_same_condition(self):
        p=self.fixture();p['evidence_documents'][1]['text']=p['evidence_documents'][1]['text'].replace('is compact','is countably compact')
        c=self.run_refs(p)['comparisons'][0]
        self.assertEqual(c['mapping'][0]['status'],'unmatched_selected_item')
        self.assertTrue(all(not m['certified_correspondence'] for m in c['mapping'][0]['possible_text_matches']))

    def test_duplicate_counter_and_wrong_endpoint_rejected(self):
        for case in ['duplicate','counter','version','missing','conditional']:
            p=self.fixture();d=p['evidence_documents'][1]
            if case=='duplicate':d['text']=d['text'].replace(r'\item[(2)]',r'\item[(1)]')
            if case=='counter':d['text']=d['text'].replace(r'\begin{itemize}',r'\setcounter{enumi}{8}\begin{itemize}')
            if case=='version':d['version']=3
            if case=='missing':d['text']=p['new']['text']
            if case=='conditional':d['text']=r'\iffalse '+d['text']
            with self.subTest(case=case):self.assertEqual(self.run_refs(p)['comparisons'],[])

    def test_nested_item_does_not_steal_parent_number(self):
        s=r'\begin{enumerate}\item A\begin{itemize}\item Nested\end{itemize}\item B\end{enumerate}'
        items=numbered_items(s,0,len(s))
        self.assertEqual(sorted(items),[1,2]);self.assertIn('Nested',items[1]['quote'])
        self.assertEqual(items[2]['quote'],'B')

    def test_wrapped_numeric_selectors(self):
        p=self.fixture()
        for name in ['old','new']:
            old=p[name]['text'];new=old.replace('(1)',r'\textbf{\blue{(1)}}')
            p[name]['text']=new
            for d in p['evidence_documents']:
                if d['side']==name:d['text']=d['text'].replace(old,new)
        self.assertEqual(len(self.run_refs(p)['comparisons']),1)

    def test_distinct_subject_branch_stays_visible_without_fake_alignment(self):
        a='Assume $X$ is compact. If $u$ is trivial, then $P$.'
        b='Assume $X$ is compact. If $F(u)=u$ or $G(u)=u$, then $P$.'
        r=unmatched_clauses(a,b,comparison(a,b))
        self.assertEqual(r['old'][0]['quote'],'If $u$ is trivial')
        self.assertEqual(r['new'][0]['quote'],'If $F(u)=u$ or $G(u)=u$')
        self.assertFalse(r['added_premise_certified'])

    def test_identical_formula_condition_not_repeated_as_unmatched(self):
        a='Assume $X$ is compact. If $F(t)=0$, then P.'
        b='Assume $Y$ is compact. If $F(t)=0$, then P.'
        r=unmatched_clauses(a,b,comparison(a,b))
        self.assertTrue(r)
        self.assertNotIn('F(t)',str(r))

    def test_formula_anchor_is_only_a_candidate_despite_prose_change(self):
        from condition_references import text_match_score,item_key
        a=r'$\E{\|x(0)\|^2+\|y(0)\|^2}<\infty$. Further explanatory text.'
        b=r'$\mathbb{E}[\|x(0)\|^2+\|y(0)\|^2]<\infty$.'
        score,basis=text_match_score(a,b,item_key(a),item_key(b))
        self.assertGreater(score,.7)
        self.assertEqual(basis,'first_formula_text_not_semantic_identity')

    def test_resolved_list_has_one_consistent_target_status(self):
        from strength_signals import analyze_pairs
        from render_results import render_markdown
        r=analyze_pairs([self.fixture()])[0]
        a=r['algorithm']
        self.assertEqual(a['reference_expansion']['endpoints']['old']['conditions']['status'],'resolved_as_numbered_condition_list')
        self.assertNotIn('未展开或有歧义的引用',render_markdown([r]))
        self.assertIn('排版归一化后字面相同',render_markdown([r]))

    def test_branch_correspondence_is_in_main_route_without_primary_upgrade(self):
        from strength_signals import analyze_pairs
        a='Let $X$ be a group acting continuously on a compact connected metric space $M$. Fix an invariant probability measure and its associated representation. If $u$ is trivial, then $P(X)$ holds for every element.'
        b=a.replace('$u$ is trivial','$F(u)=u$ or $G(u)=u$')
        r=analyze_pairs([pair(a,b)])[0]['algorithm']
        self.assertEqual(r['investigation_queue']['status'],'condition_clause_correspondence_pending')
        self.assertFalse(r['eligible_for_primary_investigation_queue'])


if __name__=='__main__':unittest.main()
