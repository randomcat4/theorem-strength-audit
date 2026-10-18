import unittest
from render_results import render_markdown, summarize_result, _location
from strength_signals import analyze_pairs


class RewritePresentation(unittest.TestCase):
    def test_recognized_rewrite_does_not_display_conflicting_mixed_change(self):
        pair = {'pair_id': 'synthetic-translation',
                'old': {'paper_id': 'example', 'version': 1, 'text': r'Let $X$ be a space. Then $X\cong Y$.'},
                'new': {'paper_id': 'example', 'version': 2, 'text': r'Let $X$ be a space. Then $X$ isomorphic to $Y$.'}}
        result = analyze_pairs([pair])[0]
        self.assertTrue(result['algorithm']['rewrite_equivalence']['suppresses_strength_alert'])
        text = render_markdown([result])
        self.assertIn('限定换写规则已覆盖', text)
        self.assertNotIn('结论或其他公式也变了，整体方向未定', text)
        self.assertEqual(summarize_result(result)['algorithm']['investigation_queue']['status'], 'rewrite_only')

    def test_original_offsets_are_not_presented_as_source_lines(self):
        text = _location({'path': 'C:/example.tex', 'span': [12000, 12345]})
        self.assertIn('不是行号', text)
        self.assertIn('12000, 12345', text)
        self.assertNotIn('节选／索引来源', text)

    def test_type_rewrite_does_not_reactivate_legacy_unclassified_signal(self):
        p={'pair_id':'type-spelling','old':{'paper_id':'p','version':1,'text':'Let $X$ be a complex manifold. Then F is transversally smooth.'},
           'new':{'paper_id':'p','version':2,'text':'Let $X$ be a smooth complex manifold. Then F is transversely smooth.'}}
        r=analyze_pairs([p])[0]
        self.assertTrue(r['algorithm']['rewrite_equivalence']['suppresses_strength_alert'])
        text=render_markdown([r])
        self.assertNotIn('有效正文有未分类变化',text)
        self.assertNotIn('未归类的可见文字差异',text)

    def test_same_abstract_two_roles_are_one_matched_source(self):
        from render_results import _compact_disclosure_evidence
        match={'doc_id':'abstract','source_type':'abstract','sha256':'one-file','quote':'We replace a claim.','source_path':'evidence.html'}
        duplicate={**match,'doc_id':'notice','source_type':'public_erratum'}
        d=_compact_disclosure_evidence({'matched_evidence':[match,duplicate]})
        self.assertEqual(len(d['matched_evidence']),1)
        self.assertEqual(d['duplicate_source_role_records_merged'],1)
        self.assertEqual(d['matched_evidence'][0]['source_roles'],['abstract','public_erratum'])
        self.assertFalse(d['matched_evidence'][0]['source_roles_are_independent_evidence'])

    def test_local_estimate_condition_mentions_formula_target(self):
        from test_strength_signals import pair,side
        a=r'For every $k$, there exists $u_k$ solving \[L u_k=f_k.\] such that for every $a>0$ \[\|u_k\|\leq C.\]'
        b=r'For every $k$, there exists $u_k$ solving \[L u_k=f_k.\] such that for every $a>0$, $k\neq0$, \[\|u_k\|\leq C.\label{estimate}\]'
        r=analyze_pairs([pair(side('p',a,version=1),side('p',b,version=2))])[0]
        text=render_markdown([r])
        self.assertIn('不能外推为整条命题的条件',text)
        self.assertIn('estimate',text)
        self.assertIn('不能拿它当旧适用域',text)


if __name__ == '__main__':
    unittest.main()
