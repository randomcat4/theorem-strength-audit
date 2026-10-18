"""Source identity, lossless recovery and reader navigation regressions."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
import urllib.request
from http.server import HTTPServer
import zipfile

from tex_sources import (CorpusStore, SourceResolver, bind_cached_version, digest,
                         locate_normalised_statement, make_handler, normalise_newlines, source_key)


class MemoryCorpus:
    def __init__(self, text):
        self.record = {'paper_id': 'paper', 'text': text, 'text_sha256': digest(text.encode()),
                       'corpus_locator': {'paper_id': 'paper', 'text_sha256': digest(text.encode())}}

    def get(self, paper_id):
        return self.record if paper_id == 'paper' else None


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = MemoryCorpus('intro\n\\begin{lemma}Body.\\end{lemma}\n')
        self.resolver = SourceResolver.__new__(SourceResolver)
        self.resolver.config = {}
        self.resolver.store = self.corpus
        self.resolver.catalog = {'sources': {}, 'version_bindings': {}, 'legacy_paths': {}}
        self.resolver.base_url = 'http://127.0.0.1:8769'

    def cached(self, text, version=1, original=None):
        path = self.root / 'expanded.tex'
        path.write_text(text, encoding='utf-8')
        if original is not None:
            project = self.root / 'project'
            project.mkdir(exist_ok=True)
            main = project / 'main.tex'
            main.write_text(original, encoding='utf-8')
            archive = self.root / 'source.bin'
            archive.write_bytes(b'synthetic original archive')
            receipt = {'archive': str(archive), 'archive_sha256': digest(archive.read_bytes()),
                       'root': 'main.tex', 'member_sha256': {'main.tex': digest(main.read_bytes())}}
            (self.root / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
            archive.with_suffix('.json').write_text(json.dumps({'version': version,
                'sha256': receipt['archive_sha256'], 'url': 'https://arxiv.org/src/paper'}), encoding='utf-8')
        return {'paper_id': 'paper', 'version': version, 'text': 'Body.', 'source': {'path': str(path)}}

    def test_unknown_snapshot_reads_original_without_materialising_file(self):
        ep = {'paper_id': 'paper', 'version': None}
        before = set(self.root.iterdir())
        access = self.resolver.resolve(ep)
        self.assertEqual(access['backend'], 'corpus')
        self.assertEqual(self.resolver.read(access), self.corpus.record['text'])
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertIsNone(access['path'])
        self.assertEqual(access['version_binding']['status'], 'snapshot_version_unknown')

    def test_statement_overlap_cannot_bind_historical_version(self):
        ep = self.cached('Old intro\nBody.\n', original='Old intro\nBody.\n')
        access = self.resolver.resolve(ep)
        self.assertEqual(access['backend'], 'historical_cache')
        self.assertNotIn(source_key(ep), self.resolver.catalog['version_bindings'])

    def test_complete_native_source_reuses_corpus_before_expanded_copy(self):
        ep = self.cached('expanded formatting\nBody.', original=self.corpus.record['text'])
        access = self.resolver.resolve(ep)
        self.assertEqual(access['backend'], 'corpus')
        self.assertEqual(access['version_binding']['version'], 1)
        self.assertEqual(self.resolver.read(access), self.corpus.record['text'])

    def test_wrong_native_version_is_not_reused(self):
        ep = self.cached('Body.', version=2, original=self.corpus.record['text'])
        ep['version'] = 1
        access = self.resolver.resolve(ep)
        self.assertEqual(access['backend'], 'missing')
        self.assertEqual(access['cached_native_version'], 2)

    def test_exact_native_bibliography_append_is_allowed(self):
        main = 'main TeX\n'
        bib = r'\begin{thebibliography}References.\end{thebibliography}'
        ep = self.cached('expanded', original=main)
        bpath = self.root / 'project/main.bbl'
        bpath.write_text(bib, encoding='utf-8')
        receipt = json.loads((self.root / 'receipt.json').read_text(encoding='utf-8'))
        receipt['member_sha256']['main.bbl'] = digest(bpath.read_bytes())
        (self.root / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
        self.corpus = MemoryCorpus(main + bib)
        self.resolver.store = self.corpus
        access = self.resolver.resolve(ep)
        self.assertEqual(access['backend'], 'corpus')
        self.assertIn('main.bbl', access['version_binding']['matched_original_members'])

    def test_unverified_corpus_suffix_cannot_be_ignored(self):
        ep = self.cached('Body.', original=self.corpus.record['text'])
        self.resolver.store = MemoryCorpus(self.corpus.record['text'] + 'new proof assumption')
        self.assertEqual(self.resolver.resolve(ep)['backend'], 'historical_cache')

    def test_reversible_native_encoding_requires_complete_equality(self):
        ep = self.cached('Body.', original='main')
        path = self.root / 'project/main.tex'
        path.write_bytes(b'old\xd0text\n')
        receipt = json.loads((self.root / 'receipt.json').read_text(encoding='utf-8'))
        receipt['member_sha256']['main.tex'] = digest(path.read_bytes())
        (self.root / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
        self.resolver.store = MemoryCorpus('old\xd0text\n')
        access = self.resolver.resolve(ep)
        self.assertEqual(access['backend'], 'corpus')
        self.assertEqual(access['version_binding']['original_decoding'], 'latin1_exact_corpus_match')

    def test_tampered_original_member_is_rejected(self):
        ep = self.cached('Body.', original=self.corpus.record['text'])
        (self.root / 'project/main.tex').write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'member hash'):
            self.resolver.resolve(ep)

    def test_missing_version_returns_missing_without_downloading(self):
        a = self.resolver.resolve({'paper_id': 'paper', 'version': 17})
        self.assertEqual(a['backend'], 'missing')
        self.assertFalse(a['fetch_performed'])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_exact_archive_recovery_retains_old_hash(self):
        data = b'original\r\nbytes\xd0\n'
        archive = self.root / 'retired.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('old.tex', data)
        a = {'backend': 'legacy_archive', 'archive_path': str(archive),
             'archive_member': 'old.tex', 'sha256': digest(data)}
        self.assertEqual(self.resolver.read_bytes(a), data)
        a['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.resolver.read(a)

    def test_navigation_overlay_keeps_algorithm_and_statement_unchanged(self):
        ep = {'paper_id': 'paper', 'version': None, 'text': 'Body.', 'source': {}}
        row = {'pair_id': 'p', 'old': ep, 'new': deepcopy(ep), 'algorithm': {'priority_score': 7, 'eligible': False}}
        prior = deepcopy(row)
        result = self.resolver.project(row)
        self.assertEqual(row, prior)
        self.assertEqual(result['algorithm'], row['algorithm'])
        self.assertEqual(result['new']['text'], row['new']['text'])
        self.assertEqual(result['new']['source']['source_access']['statement_start_line'], 2)

    def test_frozen_old_pointer_reads_exact_bytes_after_retirement(self):
        ep = self.cached('old evidence\n', original='different old source')
        access = self.resolver.resolve(ep)
        old_path = Path(access['path'])
        data = old_path.read_bytes()
        archive = self.root / 'retired.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('source.tex', data)
        self.resolver.catalog['legacy_paths'][str(old_path)] = 'retired'
        self.resolver.catalog['sources']['retired'] = {'backend': 'legacy_archive',
            'sha256': digest(data), 'archive_path': str(archive), 'archive_member': 'source.tex'}
        old_path.unlink()
        self.assertEqual(self.resolver.read_bytes(access), data)

    def test_only_newline_normalisation_is_allowed_for_version_binding(self):
        self.assertEqual(normalise_newlines('a\r\n\n'), 'a')
        self.assertNotEqual(normalise_newlines('a % note'), normalise_newlines('a'))
        self.assertNotEqual(normalise_newlines('a b'), normalise_newlines('ab'))

    def test_navigation_comment_mask_keeps_actual_source_offsets(self):
        text = 'preface\nLet $x$ % comment\n be positive.\nend'
        a, b, method = locate_normalised_statement(text, 'Let $x$ be positive.')
        self.assertEqual(text[a:b], 'Let $x$ % comment\n be positive.')
        self.assertEqual(text.count('\n', 0, a) + 1, 2)
        self.assertEqual(method, 'unique_comment_masked_whitespace_match')

    def test_navigation_does_not_merge_tex_control_word_boundaries(self):
        a, b, _ = locate_normalised_statement(r'For $x\inX$.', r'For $x\in X$.')
        self.assertIsNone(a)
        self.assertIsNone(b)

    def test_navigation_repeated_matching_statement_is_ambiguous(self):
        a, _, _ = locate_normalised_statement('Body.\nBody.', 'Body.')
        self.assertIsNone(a)

    def test_viewer_returns_escaped_full_source_and_line_anchor(self):
        text = self.corpus.record['text'] + '<script>untrusted()</script>\n'
        self.corpus = MemoryCorpus(text)
        self.resolver.store = self.corpus
        access = self.resolver.resolve({'paper_id': 'paper', 'version': None})
        server = HTTPServer(('127.0.0.1', 0), make_handler(self.resolver))
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        url = 'http://127.0.0.1:' + str(server.server_address[1]) + '/tex/' + access['ref']
        with opener.open(url + '?line=2', timeout=5) as response:
            content = response.read().decode()
        self.assertIn('id="L2"', content)
        self.assertIn('&lt;script&gt;untrusted()', content)
        self.assertNotIn('<script>untrusted()', content)
        with opener.open(url + '?raw=1', timeout=5) as response:
            self.assertEqual(response.read().decode(), text)
        with self.assertRaises(urllib.error.HTTPError):
            opener.open('http://127.0.0.1:' + str(server.server_address[1]) + '/tex/../../secret', timeout=5)


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip'), 'optional corpus reader dependency')
class RealCorpusReaderTests(unittest.TestCase):
    def test_existing_seek_index_read_and_integrity_gate(self):
        import indexed_gzip
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = {'paper_id': 'paper', 'text': 'complete TeX\n'}
            data['text_sha256'] = digest(data['text'].encode())
            raw = (json.dumps(data) + '\n').encode()
            shard = root / 'one.gz'
            with gzip.open(shard, 'wb') as stream:
                stream.write(raw)
            with indexed_gzip.IndexedGzipFile(filename=str(shard)) as stream:
                stream.build_full_index()
                stream.export_index(filename=str(root / 'one.gz.gzidx'))
            connection = sqlite3.connect(root / 'locators.sqlite')
            connection.execute('CREATE TABLE paper(paper_id TEXT PRIMARY KEY,shard TEXT,offset INTEGER,length INTEGER,row INTEGER)')
            connection.execute('INSERT INTO paper VALUES(?,?,?,?,?)', ('paper', 'one.gz', 0, len(raw), 0))
            connection.commit(); connection.close()
            before = digest(shard.read_bytes())
            reader = CorpusStore(root, root / 'locators.sqlite', root)
            self.assertEqual(reader.get('paper')['text'], data['text'])
            self.assertIsNone(reader.get('missing'))
            self.assertEqual(digest(shard.read_bytes()), before)
            self.assertEqual(set(p.name for p in root.iterdir()), {'one.gz', 'one.gz.gzidx', 'locators.sqlite'})


if __name__ == '__main__':
    unittest.main()
