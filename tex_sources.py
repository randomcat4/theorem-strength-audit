"""Read original corpus rows first; keep historical evidence separate.

The corpus, its existing seek indexes and all result files are read-only. No
index builder, network fetcher or model client lives in this module. A local
HTTP viewer makes compressed corpus text clickable without materialising TeX.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import closing
from copy import deepcopy
import hashlib
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qs, quote, urlsplit
import zipfile

VERSION = '1.0.0'
MAX_ROW_BYTES = 32 * 1024 * 1024
MAX_TEX_BYTES = 8 * 1024 * 1024
MAX_CACHE_BYTES = 16 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalise_newlines(text):
    """Only newline encoding and trailing empty lines; no TeX rewriting."""
    return text.replace('\r\n', '\n').replace('\r', '\n').rstrip('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def locate_normalised_statement(text, body):
    """Unique literal match after comment masking/whitespace folding, with real offsets.

    This is navigation only, never evidence that two versions are identical.
    Control-word separators are retained; no macro expansion or token sorting.
    """
    if len(text.encode('utf-8')) > MAX_TEX_BYTES or len(body) > 100000:
        return None, None, 'not_located_in_full_source'
    if re.search(r'\\verb\b|\\begin\s*\{(?:verbatim|lstlisting|minted)', body):
        return None, None, 'not_located_in_full_source'
    comments = re.compile(r'(?<!\\)((?:\\\\)*)%[^\r\n]*')
    def mask(value):
        return comments.sub(lambda m: m[1] + ' ' * (len(m[0])-len(m[1])), value)
    masked = mask(text)
    folded = re.sub(r'\s+', ' ', masked)
    needle = re.sub(r'\s+', ' ', mask(body)).strip()
    if not needle:
        return None, None, 'not_located_in_full_source'
    start = folded.find(needle)
    if start < 0 or folded.find(needle, start + 1) >= 0:
        return None, None, 'not_located_in_full_source'
    end = start + len(needle)
    cursor, raw_start, raw_end = 0, None, None
    for match in re.finditer(r'\s+|\S+', masked):
        whitespace = match[0].isspace()
        width = 1 if whitespace else len(match[0])
        if raw_start is None and cursor <= start < cursor + width:
            raw_start = match.start() if whitespace else match.start() + start - cursor
        if cursor < end <= cursor + width:
            raw_end = match.end() if whitespace else match.start() + end - cursor
            break
        cursor += width
    if raw_start is None or raw_end is None:
        return None, None, 'not_located_in_full_source'
    return raw_start, raw_end, 'unique_comment_masked_whitespace_match'


def source_key(ep):
    return str(ep['paper_id']) + '@' + str(ep.get('version'))


def bind_cached_version(path, version, corpus):
    """Bind native version metadata using complete original main-source equality."""
    path = Path(path)
    receipt_path = path.parent / 'receipt.json'
    if not receipt_path.is_file():
        return None
    receipt = read_json(receipt_path)
    archive = Path(receipt.get('archive', ''))
    download_receipt = archive.with_suffix('.json')
    if not download_receipt.is_file() or not archive.is_file():
        return None
    native = read_json(download_receipt)
    if native.get('version') != version or native.get('sha256') != receipt.get('archive_sha256'):
        return None
    if not re.fullmatch(r'https://arxiv\.org/(?:src|e-print)/' + re.escape(corpus['paper_id']) + r'(?:v\d+)?', native.get('url', '')):
        return None
    if archive.stat().st_size > MAX_ROW_BYTES or digest(archive.read_bytes()) != receipt['archive_sha256']:
        raise ValueError('Native source archive size or hash changed')
    root = receipt.get('root')
    if not root:
        return None
    project = (path.parent / 'project').resolve()
    main = (project / root).resolve()
    if not main.is_relative_to(project):
        raise ValueError('Original source member outside cached project')
    if not main.is_file() or main.stat().st_size > MAX_TEX_BYTES:
        return None
    data = main.read_bytes()
    if digest(data) != receipt.get('member_sha256', {}).get(root):
        raise ValueError('Original source member hash changed')
    def decode(raw):
        try:
            return raw.decode('utf-8'), 'utf8'
        except UnicodeDecodeError:
            # A reversible byte mapping; accepted only after whole-text equality.
            return raw.decode('latin1'), 'latin1_exact_corpus_match'
    main_text, encoding = decode(data)
    target = normalise_newlines(corpus['text'])
    original = normalise_newlines(main_text)
    used_members = {root: digest(data)}
    method = 'complete_original_main_source_equality'
    if original != target:
        # The original dataset sometimes appends a .bbl after the main source.
        # Accept only complete, hash-verified native bibliography members. An
        # arbitrary suffix, proof change or whitespace rewrite is not accepted.
        if not target.startswith(original):
            return None
        remaining = target[len(original):].lstrip('\n')
        for name, expected in sorted(receipt.get('member_sha256', {}).items()):
            if not name.endswith('.bbl'):
                continue
            member = (project / name).resolve()
            if not member.is_relative_to(project) or not member.is_file() or member.stat().st_size > MAX_TEX_BYTES:
                return None
            content = member.read_bytes()
            if digest(content) != expected:
                raise ValueError('Original bibliography member hash changed')
            text, _ = decode(content)
            text = normalise_newlines(text).lstrip('\n')
            if text and remaining.startswith(text):
                remaining = remaining[len(text):].lstrip('\n')
                used_members[name] = expected
        if remaining or len(used_members) == 1:
            return None
        method = 'complete_main_plus_exact_native_bibliography_members'
    return {'status': 'native_version_bound_by_complete_original_source',
        'version': version, 'corpus_text_sha256': corpus['text_sha256'],
        'requested_version_tex_content_verified': True, 'corpus_native_snapshot_version_uniquely_identified': False,
        'method': method, 'newline_only_normalisation': True, 'original_decoding': encoding,
        'matched_original_members': used_members,
        'native_archive': str(archive), 'native_archive_sha256': receipt['archive_sha256'],
        'native_receipt': str(download_receipt), 'native_receipt_sha256': digest(download_receipt.read_bytes()),
        'original_member': str(main), 'original_member_sha256': digest(data)}


class CorpusStore:
    def __init__(self, body_root, locator_db, index_root):
        self.body_root = Path(body_root).resolve()
        self.index_root = Path(index_root).resolve()
        self.locator_db = Path(locator_db).resolve()
        self.cache = OrderedDict()
        self.cache_bytes = 0

    def get(self, paper_id):
        if paper_id in self.cache:
            self.cache.move_to_end(paper_id)
            return self.cache[paper_id]
        uri = self.locator_db.as_uri() + '?mode=ro'
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.execute('PRAGMA cache_size=-2048')
            row = connection.execute('SELECT shard,offset,length,row FROM paper WHERE paper_id=?', (paper_id,)).fetchone()
        if row is None:
            return None
        shard, offset, length, row_number = row
        if Path(shard).name != shard or not isinstance(length, int) or not 0 < length <= MAX_ROW_BYTES:
            raise ValueError('Unsafe or oversized corpus locator')
        if not isinstance(offset, int) or offset < 0:
            raise ValueError('Invalid corpus offset')
        # Deliberately lazy: the standalone detector has no gzip dependency.
        import indexed_gzip
        with indexed_gzip.IndexedGzipFile(filename=str(self.body_root / shard),
                                         spacing=64 * 1024**2, readbuf_size=1024**2) as stream:
            stream.import_index(filename=str(self.index_root / (shard + '.gzidx')))
            stream.seek(offset)
            raw = stream.read(length)
        record = json.loads(raw)
        text = record['text']
        data = text.encode('utf-8')
        if record['paper_id'] != paper_id or digest(data) != record['text_sha256']:
            raise ValueError('Corpus paper identity or text hash mismatch')
        if len(data) > MAX_TEX_BYTES:
            raise ValueError('Corpus TeX exceeds bounded full-text limit')
        record['corpus_locator'] = {'container_path': str(self.body_root / shard),
            'index_path': str(self.index_root / (shard + '.gzidx')),
            'row': row_number, 'uncompressed_json_offset': offset, 'json_bytes': length,
            'paper_id': paper_id, 'text_sha256': digest(data)}
        while self.cache and self.cache_bytes + len(data) > MAX_CACHE_BYTES:
            _, previous = self.cache.popitem(last=False)
            self.cache_bytes -= len(previous['text'].encode('utf-8'))
        if len(data) <= MAX_CACHE_BYTES:
            self.cache[paper_id] = record
            self.cache_bytes += len(data)
        return record


class SourceResolver:
    def __init__(self, config):
        self.config = config
        self.store = CorpusStore(**config['corpus'])
        self.catalog = read_json(config['catalog']) if config.get('catalog') and Path(config['catalog']).exists() else {}
        self.base_url = config.get('base_url', 'http://127.0.0.1:8769').rstrip('/')

    @classmethod
    def from_file(cls, path):
        return cls(read_json(path))

    def url(self, ref, line=None):
        result = self.base_url + '/tex/' + quote(ref, safe='')
        if line:
            result += '?line=' + str(int(line)) + '#L' + str(int(line))
        return result

    def read_bytes(self, access):
        # Frozen result JSON and earlier compact cards retain their old paths.
        # Resolve those pointers losslessly rather than requiring a rewritten
        # mathematical result or silently substituting different source bytes.
        ref = access.get('ref')
        path = access.get('path')
        if not ref and path:
            ref = self.catalog.get('legacy_paths', {}).get(str(Path(path).resolve()))
        registered = self.catalog.get('sources', {}).get(ref)
        if registered and registered.get('sha256') == access.get('sha256'):
            access = registered
        backend = access.get('backend')
        if backend == 'corpus':
            record = self.store.get(access['paper_id'])
            if record is None or record['text_sha256'] != access['sha256']:
                raise ValueError('Original corpus source changed or missing')
            return record['text'].encode('utf-8')
        if backend == 'legacy_archive':
            with zipfile.ZipFile(access['archive_path']) as archive:
                info = archive.getinfo(access['archive_member'])
                if info.file_size > MAX_TEX_BYTES:
                    raise ValueError('Archived TeX exceeds source limit')
                data = archive.read(info)
        else:
            path = Path(access['path'])
            if path.stat().st_size > MAX_TEX_BYTES:
                raise ValueError('Cached TeX exceeds source limit')
            data = path.read_bytes()
        if digest(data) != access['sha256']:
            raise ValueError('Source evidence hash changed')
        return data

    def read(self, access):
        return self.read_bytes(access).decode('utf-8', errors='replace')

    def resolve(self, ep):
        """Versioned endpoints need a full-source binding, never statement overlap."""
        record = self.store.get(ep['paper_id'])
        version = ep.get('version')
        binding = self.catalog.get('version_bindings', {}).get(source_key(ep))
        old = ep.get('source', {}).get('source_access', {})
        path = old.get('path') or ep.get('source', {}).get('path')
        if record is not None and version is not None and not binding and path:
            binding = bind_cached_version(path, version, record)
            if binding:
                self.catalog.setdefault('version_bindings', {})[source_key(ep)] = binding
        if record is not None and (version is None or (binding and binding['corpus_text_sha256'] == record['text_sha256'])):
            access = {'backend': 'corpus', 'paper_id': ep['paper_id'], 'version': version,
                'sha256': record['text_sha256'], 'bytes': len(record['text'].encode('utf-8')),
                'corpus_locator': record['corpus_locator'], 'path': None,
                'full_tex_available': True, 'source_scope': 'original_corpus_text_read_in_place',
                'statement_match_scope': 'original_corpus_text', 'source_encoding': 'utf8',
                'version_binding': binding or {'status': 'snapshot_version_unknown'},
                'ref': 'corpus-' + digest((source_key(ep) + record['text_sha256']).encode())[:24]}
            access['viewer_url'] = self.url(access['ref'])
            self.catalog.setdefault('sources', {})[access['ref']] = access
            return access
        if version is not None and path:
            receipt_path = Path(path).parent / 'receipt.json'
            if receipt_path.is_file():
                receipt = read_json(receipt_path)
                native_path = Path(receipt.get('archive', '')).with_suffix('.json') if receipt.get('archive') else None
                if native_path and native_path.is_file():
                    native = read_json(native_path)
                    if native.get('version') is not None and native['version'] != version:
                        return {'backend': 'missing', 'full_tex_available': False, 'paper_id': ep['paper_id'],
                            'version': version, 'path': None, 'fetch_performed': False,
                            'reason': 'existing_cache_native_version_conflicts_with_requested_version',
                            'cached_native_version': native['version']}
        alias = self.catalog.get('legacy_paths', {}).get(str(Path(path).resolve())) if path else None
        if alias:
            access = dict(self.catalog['sources'][alias])
        elif path and Path(path).is_file() and Path(path).suffix == '.tex':
            p = Path(path)
            if p.stat().st_size > MAX_TEX_BYTES:
                raise ValueError('Cached TeX exceeds source limit')
            data = p.read_bytes()
            access = {'backend': 'historical_cache', 'path': str(p.resolve()), 'sha256': digest(data),
                'bytes': len(data), 'paper_id': ep['paper_id'], 'version': version,
                'source_scope': 'cached_version_source_corpus_version_not_verified',
                'full_tex_available': True}
        else:
            return {'backend': 'missing', 'full_tex_available': False, 'paper_id': ep['paper_id'],
                'version': version, 'path': None,
                'reason': 'corpus_missing_requested_version_and_no_existing_version_cache',
                'fetch_performed': False, 'requested_version': version}
        access.update(paper_id=ep['paper_id'], version=version)
        if access.get('ref'):
            access['viewer_url'] = self.url(access['ref'])
        access['corpus_fallback_reason'] = 'paper_not_in_corpus' if record is None else 'requested_version_not_bound_to_corpus_snapshot'
        return access

    def save_catalog(self):
        """Persist pointers only, never materialised corpus text."""
        if self.config.get('catalog'):
            destination = Path(self.config['catalog'])
            temporary = destination.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            temporary.replace(destination)

    def project(self, row):
        """Source-navigation overlay only. The frozen algorithm object is copied intact."""
        result = deepcopy(row)
        from prepare_recheck99 import locate
        for side in ('old', 'new'):
            ep = result[side]
            access = self.resolve(ep)
            if access.get('full_tex_available'):
                text = self.read(access).replace('\r\n', '\n')
                start, end, method = locate(text, ep['text'].replace('\r\n', '\n'))
                access.update(match_status=method, statement_start_line=None, statement_end_line=None)
                if start is not None:
                    access.update(statement_start_line=text.count('\n', 0, start) + 1,
                                  statement_end_line=text.count('\n', 0, end) + 1)
                if access.get('ref'):
                    access['viewer_url'] = self.url(access['ref'], access['statement_start_line'])
            ep.setdefault('source', {})['source_access'] = access
            original = ep['source'].get('path')
            alias = self.catalog.get('legacy_paths', {}).get(str(Path(original).resolve())) if original else None
            if alias:
                ep['source']['viewer_url'] = self.url(alias)
        # Scans were performed on the old bytes. Keep their hashes and text;
        # point to those exact archived bytes, never relabel them as corpus scans.
        for doc in result.get('evidence_documents', []):
            p = doc.get('source_path')
            alias = self.catalog.get('legacy_paths', {}).get(str(Path(p).resolve())) if p else None
            if alias:
                doc['viewer_url'] = self.url(alias)
                doc['source_storage'] = self.catalog['sources'][alias]
        assert result.get('algorithm') == row.get('algorithm')
        return result


_default = None


def default_resolver():
    global _default
    config = Path(os.environ.get('THEOREM_SOURCE_CONFIG', Path(__file__).with_name('source_config.local.json')))
    if not config.is_file():
        return None
    if _default is None:
        _default = SourceResolver.from_file(config)
    return _default


def project_result(row):
    resolver = default_resolver()
    return resolver.project(row) if resolver else row


def read_source(access):
    resolver = default_resolver()
    if resolver:
        return resolver.read(access)
    p = Path(access['path'])
    if p.stat().st_size > MAX_TEX_BYTES:
        raise ValueError('Source limit exceeded')
    data = p.read_bytes()
    if access.get('sha256') and digest(data) != access['sha256']:
        raise ValueError('Source hash mismatch')
    return data.decode('utf-8', errors='replace')


def access_link(access, label, line=None):
    url = access.get('viewer_url')
    if url:
        if line:
            url = url.split('?')[0].split('#')[0] + '?line=' + str(line) + '#L' + str(line)
        return '[' + label + '](<' + url + '>)'
    value = str(access.get('path', '')).replace('\\', '/')
    if line:
        value += ':' + str(line)
    return '[' + label + '](<' + value + '>)'


def make_handler(resolver):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Adapters may register new pointers while this read-only viewer runs.
            if resolver.config.get('catalog'):
                resolver.catalog = read_json(resolver.config['catalog'])
            parsed = urlsplit(self.path)
            if parsed.path == '/health':
                body = json.dumps({'status': 'ok', 'service': 'theorem-source-reader', 'version': VERSION,
                                   'catalog_sources': len(resolver.catalog.get('sources', {}))}).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(body)
                return
            ref = parsed.path.removeprefix('/tex/')
            access = resolver.catalog.get('sources', {}).get(ref) if parsed.path.startswith('/tex/') else None
            if access is None:
                self.send_error(404); return
            try:
                text = resolver.read(access)
                line = int(parse_qs(parsed.query).get('line', ['1'])[0])
                if line < 1:
                    raise ValueError('invalid line')
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                self.send_error(409, str(exc)); return
            if 'raw' in parse_qs(parsed.query):
                body = resolver.read_bytes(access)
                content_type = 'text/plain; charset=utf-8'
            else:
                label = '原库完整 TeX · 按需直读' if access['backend'] == 'corpus' else '历史版本/旧扫描证据 · 缓存'
                title = html.escape(str(access.get('paper_id', '')) + ' ' + str(access.get('version') or '') + ' · ' + label)
                rows = '\n'.join('<div id="L{0}" class="line"><a href="#L{0}">{0}</a> {1}</div>'.format(i, html.escape(s))
                                 for i, s in enumerate(text.splitlines(), 1))
                body = ('<!doctype html><meta charset="utf-8"><title>' + title + '</title>'
                    '<style>body{margin:24px;font:15px/1.55 monospace;background:#fafaf8;color:#222}'
                    '.line{white-space:pre-wrap;overflow-wrap:anywhere}.line a{color:#777;text-decoration:none}'
                    '.line:target{background:#fff0a6}header{font-family:sans-serif;margin-bottom:20px}</style>'
                    '<header><h2>' + title + '</h2><p>SHA-256: ' + access['sha256'] + '</p>'
                    '<p>行号属于此来源全文；原库压缩分片未修改，没有生成新的 TeX 文件。</p>'
                    '<a href="?raw=1">读取原始 TeX 文本</a></header>' + rows +
                    '<script>document.getElementById("L' + str(line) + '")?.scrollIntoView({block:"center"});</script>').encode('utf-8')
                content_type = 'text/html; charset=utf-8'
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('serve')
    read = sub.add_parser('read')
    read.add_argument('--ref', required=True)
    read.add_argument('--start-line', type=int, default=1)
    read.add_argument('--end-line', type=int)
    args = parser.parse_args()
    resolver = SourceResolver.from_file(args.config)
    if args.command == 'serve':
        address = urlsplit(resolver.base_url)
        if address.hostname != '127.0.0.1':
            raise ValueError('Viewer must bind to loopback')
        HTTPServer(('127.0.0.1', address.port or 8769), make_handler(resolver)).serve_forever()
    else:
        access = resolver.catalog['sources'][args.ref]
        lines = resolver.read(access).splitlines()
        if args.start_line < 1 or (args.end_line and args.end_line < args.start_line):
            parser.error('invalid line range')
        for i in range(args.start_line - 1, min(args.end_line or len(lines), len(lines))):
            print(f'{i+1}: {lines[i]}')


if __name__ == '__main__':
    main()
