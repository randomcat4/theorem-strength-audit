import tempfile
import unittest
from pathlib import Path
from prepare_recheck99 import locate, parse_abs


class CacheAdapterTests(unittest.TestCase):
    def test_comments_class_has_extra_mathjax_token(self):
        raw = '''<h1 class="title mathjax">Title:Example</h1>
        <td class="tablecell label">Comments:</td>
        <td class="tablecell comments mathjax">Missing assumption is now corrected.</td>
        <div class="submission-history">[v1] Tue, 5 Jul 2016 09:28:08 UTC</div>'''
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'page.html'
            path.write_text(raw, encoding='utf-8')
            meta = parse_abs(raw, '0000.00000', 'https://example.invalid', path)
        self.assertTrue(meta['parsed'])
        self.assertIn('Missing assumption', meta['comments'])
        self.assertEqual(meta['versions'][1], '2016-07-05T09:28:08+00:00')

    def test_present_unparsed_comments_is_missing_evidence(self):
        raw = '''<h1 class="title">Example</h1><td>Comments:</td>
        <div class="submission-history">[v1] Tue, 5 Jul 2016 09:28:08 UTC</div>'''
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'page.html'
            path.write_text(raw, encoding='utf-8')
            self.assertFalse(parse_abs(raw, 'x', 'url', path)['parsed'])

    def test_duplicate_labels_do_not_claim_unique_source_location(self):
        text = r'\label{x} first \label{x} second'
        self.assertEqual(locate(text, r'\label{x} missing body')[2], 'not_located_in_full_source')

    def test_unique_label_returns_enclosing_statement(self):
        text = 'intro\n' + r'\begin{lemma}\label{x} Full text.\end{lemma}' + '\nproof'
        start, end, method = locate(text, r'\label{x} indexed different spacing')
        self.assertEqual(text[start:end], r'\begin{lemma}\label{x} Full text.\end{lemma}')
        self.assertEqual(method, 'unique_label_enclosing_environment')


if __name__ == '__main__':
    unittest.main()
