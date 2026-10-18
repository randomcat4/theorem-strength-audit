"""Bounded cache adapter: replay the prior automatically eligible cohort only.

Collects source material, not human verdicts. No retrieval/index rebuild, model,
or annotation input. Public pages are point requests with an immutable cache.
"""
import argparse
import collections
import datetime as dt
import hashlib
import html
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from tex_reference_keys import statement_reference
from static_tex_context import extract_context
from tex_sources import default_resolver, locate_normalised_statement

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'recheck_99_20261018'
CAP = 8 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def selected():
    cohorts = {}
    for name, input_name in [('forward', 'forward96.full.jsonl'), ('versions', 'versions40.jsonl')]:
        prior_path = HERE / 'round_4' / name / 'results.jsonl'
        prior = {r['pair_id']: r for r in jsonl(prior_path) if r['algorithm']['eligible_for_strength_queue']}
        ids = set(prior)
        rows = [r for r in jsonl(HERE / 'inputs' / input_name) if r['pair_id'] in ids]
        assert len(rows) == len(ids) == (2 if name == 'forward' else 97)
        prior_hash = hash_file(prior_path)
        frame_hash = hash_file(HERE / 'inputs' / input_name)
        for row in rows:
            row['cached_alignment_competition'] = {
                'competition': prior[row['pair_id']]['algorithm']['alignment_evidence'].get('competition', {}),
                'source_results_path': str(prior_path), 'source_results_sha256': prior_hash,
                'frame_input_path': str(HERE / 'inputs' / input_name), 'frame_input_sha256': frame_hash,
                'frame_hash': frame_hash, 'frame_name': name + '_round4_complete_cached_frame',
                'statement_alignment': prior[row['pair_id']]['algorithm']['alignment_evidence'].get('status'),
                'scope': 'full_prior_cached_frame_competition_not_recomputed_from_selected_99',
                'engine_version': '0.2.2', 'code_version': '0.2.2', 'human_or_llm_verdict': False}
        cohorts[name] = rows
    return cohorts


def plain(value):
    value = re.sub(r'<(?:script|style)\b[^>]*>.*?</(?:script|style)>', '', value, flags=re.S | re.I)
    value = re.sub(r'<[^>]+>', ' ', value)
    return re.sub(r'\s+', ' ', html.unescape(value)).strip()


def section(raw, pattern):
    m = re.search(pattern, raw, re.I | re.S)
    return plain(m.group(1)) if m else ''


def parse_abs(raw, pid, url, path):
    # arXiv's displayed version identifier binds comments to a particular page.
    abstract = section(raw, r'<blockquote[^>]*class=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</blockquote>')
    abstract = re.sub(r'^Abstract:\s*', '', abstract)
    comments = section(raw, r'<td[^>]*class=["\'][^"\']*\bcomments\b[^"\']*["\'][^>]*>(.*?)</td>')
    title = section(raw, r'<h1[^>]*class=["\'][^"\']*title[^"\']*["\'][^>]*>(.*?)</h1>')
    title = re.sub(r'^Title:\s*', '', title)
    history = section(raw, r'<div[^>]*class=["\']submission-history["\'][^>]*>(.*?)</div>')
    versions = {}
    for ver, value in re.findall(r'\[v(\d+)\]\s*((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s*\d{1,2}\s+\w+\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+\w+)', history):
        try:
            versions[int(ver)] = dt.datetime.strptime(value, '%a, %d %b %Y %H:%M:%S %Z').isoformat() + '+00:00'
        except ValueError:
            pass
    # Link /abs/idvN near "this version" or the canonical citation version.
    displayed = [int(v) for v in re.findall(re.escape(pid) + r'v(\d+)', raw)]
    has_comments_row = bool(re.search(r'<td[^>]*>\s*Comments:', raw, re.I))
    return {'paper_id': pid, 'url': url, 'source_path': str(path), 'sha256': sha(path.read_bytes()),
            'title': title, 'abstract': abstract, 'comments': comments,
            'versions': versions, 'displayed_version': max(displayed) if displayed else None,
            'comments_field_present': has_comments_row, 'comments_parse_ok': bool(comments) if has_comments_row else True,
            'parsed': bool(title and versions and (comments or not has_comments_row)),
            'retrieved_scope': 'official_current_abstract_page_not_all_historical_comments'}


def fetch_metadata(pids, fetch=False, cache_dir=None):
    cache = Path(cache_dir) if cache_dir else OUT / 'evidence_cache' / 'arxiv_abs'
    cache.mkdir(parents=True, exist_ok=True)
    records = {}
    for pid in sorted(pids):
        path = cache / (pid.replace('/', '_') + '.html')
        url = 'https://arxiv.org/abs/' + pid
        if not path.exists() and fetch:
            request = urllib.request.Request(url, headers={'User-Agent': 'Research-source-verification/1.0 (bounded arXiv abstract requests)'})
            try:
                with urllib.request.urlopen(request, timeout=25) as response:
                    data = response.read(CAP + 1)
                if len(data) > CAP:
                    raise ValueError('abstract page exceeded 8MiB cap')
                path.write_bytes(data)
            except Exception as exc:
                records[pid] = {'paper_id': pid, 'source_available': False, 'url': url, 'failure': type(exc).__name__ + ': ' + str(exc)}
                print(json.dumps({'paper_id': pid, 'metadata': 'unavailable', 'reason': str(exc)}, ensure_ascii=False), flush=True)
            time.sleep(3.1)
        if path.is_file():
            raw = path.read_bytes().decode('utf-8', errors='replace')
            records[pid] = parse_abs(raw, pid, url, path)
            records[pid]['source_available'] = records[pid]['parsed']
            print(json.dumps({'paper_id': pid, 'metadata': records[pid]['parsed'], 'versions': records[pid]['versions']}, ensure_ascii=False), flush=True)
        elif pid not in records:
            records[pid] = {'paper_id': pid, 'source_available': False, 'url': url, 'failure': 'not_in_cache'}
    save(OUT / 'evidence_cache' / 'official_metadata.json', records)
    return records


def locate(text, body):
    # Readable line pointers are derived from actual source, never indexed offsets.
    start = text.find(body)
    if start >= 0 and text.find(body, start + 1) < 0:
        return start, start + len(body), 'unique_exact_body'
    labels = re.findall(r'\\label\s*\{([^{}]+)\}', body)
    if labels:
        matches = list(re.finditer(r'\\label\s*\{\s*' + re.escape(labels[0]) + r'\s*\}', text))
        if len(matches) == 1:
            anchor = matches[0].start()
            starts = list(re.finditer(r'\\begin\{(theorem|lemma|proposition|corollary|thm|lem|prop|cor)\*?\}', text[:anchor]))
            if starts:
                opening = starts[-1]
                close = re.search(r'\\end\{' + opening.group(1) + r'\*?\}', text[anchor:])
                if close:
                    return opening.start(), anchor + close.end(), 'unique_label_enclosing_environment'
            return anchor, anchor + len(matches[0].group()), 'unique_label_only_body_extent_unknown'
    return locate_normalised_statement(text, body)


def source_access(ep):
    info = ep.setdefault('source', {})
    resolver = default_resolver()
    resolved = resolver.resolve(ep) if resolver else None
    path_value = (resolved or {}).get('path') if resolved else info.get('path')
    path = Path(path_value or '').resolve()
    access = {'path': str(path), 'full_tex_available': False, 'statement_start_line': None,
              'statement_end_line': None, 'match_status': 'missing_or_not_tex', 'source_scope': 'unknown'}
    if resolved and not resolved.get('full_tex_available'):
        info['source_access'] = resolved
        return '', resolved
    if not resolved and (not path.is_file() or path.suffix != '.tex' or path.stat().st_size > CAP):
        info['source_access'] = access
        return '', access
    if resolved:
        text = resolver.read(resolved).replace('\r\n', '\n')
        data = text.encode('utf-8')
        access = dict(resolved)
    else:
        data = path.read_bytes()
        text = data.decode('utf-8', errors='replace').replace('\r\n', '\n')
    start, end, method = locate(text, ep['text'].replace('\r\n', '\n'))
    access.update(full_tex_available=True, sha256=access.get('sha256') or sha(data), bytes=access.get('bytes') or len(data), match_status=method,
                  source_scope=access.get('source_scope') if resolved else 'complete_cached_expanded_tex_file_not_executed_tex', project_root=None,
                  related_roots=[], statement_match_scope='actual_cached_file', source_encoding='utf8_replace')
    if start is not None:
        ep['static_tex_context'] = extract_context(text, start)
        access.update(statement_start_line=text.count('\n', 0, start) + 1,
                      statement_end_line=text.count('\n', 0, end) + 1)
        reference = statement_reference(text, start)
        ep['statement_reference_keys'] = reference['keys']
        ep['statement_reference_basis'] = reference
    receipt_path = path.parent / 'receipt.json'
    if path_value and receipt_path.is_file():
        receipt = read_json(receipt_path)
        access.update(project_root=str(path.parent / 'project'), assembly_receipt=str(receipt_path),
                      assembly_receipt_sha256=sha(receipt_path.read_bytes()), related_roots=receipt.get('root_inspection', {}).get('result', {}).get('files', []),
                      selected_project_root=receipt.get('root'), unresolved_include_syntax=receipt.get('unresolved_include_syntax', []),
                      static_assembly_status=receipt.get('status'))
    elif not resolved:
        access['source_scope'] = 'complete_cached_concatenated_tex_file_active_root_requires_attention'
    access['document_roots'] = text.count('\\begin{document}')
    if resolved and access.get('ref'):
        access['viewer_url'] = resolver.url(access['ref'], access.get('statement_start_line'))
    info['source_access'] = access
    return text, access


def document(kind, ep, side, text, path, url=None, version=None, coverage=None, available=True, source_access=None):
    raw = Path(path).read_bytes() if path and Path(path).is_file() and Path(path).stat().st_size <= CAP else b''
    return {'doc_id': sha((kind + ep['paper_id'] + str(version) + side + str(path)).encode())[:24],
            'source_type': kind, 'paper_id': ep['paper_id'], 'version': version, 'side': side,
            'text': text, 'source_path': str(path) if path else None, 'url': url,
            'sha256': source_access.get('sha256') if source_access else sha(raw) if raw else None, 'coverage_scope': coverage,
            'viewer_url': source_access.get('viewer_url') if source_access else None,
            'source_storage': source_access if source_access else None,
            'source_available': available, 'identity_basis': 'paper_id_and_version_metadata_not_statement_identity',
            'association_scope': 'paper', 'statement_reference_keys': [], 'source_char_count': len(text),
            'text_truncated': False}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch-metadata', action='store_true')
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--input', type=Path, help='Optional arbitrary existing bounded candidate JSONL; omitting it selects the frozen old99.')
    parser.add_argument('--output-dir', type=Path, help='Fresh evidence output directory; default is the frozen-cohort workspace.')
    parser.add_argument('--official-cache-dir', type=Path, help='Reuse already saved official abstract HTML pages without network calls.')
    parser.add_argument('--max-pairs', type=int, default=1000)
    parser.add_argument('--source-config', type=Path, help='Original corpus/seek-index configuration; no index build or source download.')
    args = parser.parse_args()
    if args.source_config:
        os.environ['THEOREM_SOURCE_CONFIG'] = str(args.source_config.resolve())
    if args.output_dir:
        OUT = args.output_dir.resolve()
    if not 1 <= args.max_pairs <= 10000:
        parser.error('--max-pairs must be between 1 and 10000')
    provided_info = None
    if args.input:
        from run_algorithm import _read_pairs
        provided, provided_info = _read_pairs(args.input.resolve(), args.max_pairs)
        cohorts = {'provided': provided}
    else:
        cohorts = selected()
    pids = {ep['paper_id'] for rows in cohorts.values() for row in rows for ep in (row['old'], row['new'])}
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = fetch_metadata(pids, args.fetch_metadata, args.official_cache_dir)
    if args.metadata_only:
        return
    stats = collections.Counter()
    sources = {}
    public_registry = OUT / 'evidence_cache' / 'public_errata.json'
    errata = read_json(public_registry) if public_registry.exists() else []
    for cohort, rows in cohorts.items():
        path = OUT / 'inputs' / (cohort + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            for pair in rows:
                docs = []
                for side in ('old', 'new'):
                    ep = pair[side]
                    text, access = source_access(ep)
                    stats['endpoints'] += 1
                    stats['full_tex_available'] += bool(access['full_tex_available'])
                    stats['line_located'] += access['statement_start_line'] is not None
                    sources[(pair['pair_id'], side)] = {'pair_id': pair['pair_id'], 'side': side, 'paper_id': ep['paper_id'],
                                                         'statement_id': ep['statement_id'], **access}
                    if text:
                        docs.append(document('full_tex', ep, side, text, access.get('path'), version=ep.get('version'),
                                             coverage=access['source_scope'], source_access=access))
                    meta = metadata.get(ep['paper_id'], {})
                    versions = meta.get('versions', {})
                    version = ep.get('version')
                    if version is not None and str(version) in versions:
                        ep.update(date=versions[str(version)], date_kind='official_arxiv_version_history_utc', date_source=meta['url'])
                    elif version is not None and version in versions:
                        ep.update(date=versions[version], date_kind='official_arxiv_version_history_utc', date_source=meta['url'])
                    # Only the new endpoint's current page is scanned as disclosure evidence.
                    if side == 'new':
                        for kind, field in [('version_comment', 'comments'), ('abstract', 'abstract')]:
                            docs.append(document(kind, ep, side, meta.get(field, ''), meta.get('source_path'), meta.get('url'),
                                                 meta.get('displayed_version'), meta.get('retrieved_scope', 'unavailable'),
                                                 bool(meta.get('source_available'))))
                        # Public notices may be titled a note rather than "Erratum".
                        # Extraction is lexical and paper-scoped; the core must bind
                        # the affected statement itself before using it as explained.
                        abstract_notice = meta.get('abstract', '')
                        if re.search(r'\b(?:we\s+(?:retract|correct|substitute)|erratum|corrigendum|for a weaker claim)\b', abstract_notice, re.I):
                            notice = document('public_erratum', ep, side, abstract_notice, meta.get('source_path'), meta.get('url'),
                                              meta.get('displayed_version'), 'public_abstract_explicit_correction_notice_not_exhaustive_errata_search',
                                              bool(meta.get('source_available')))
                            notice['notice_kind'] = 'public_correction_notice_formal_erratum_title_not_required'
                            docs.append(notice)
                        for err in errata:
                            if ep['paper_id'] in err.get('related_paper_ids', []):
                                docs.append({**err, 'side': side, 'source_type': 'public_erratum', 'association_scope': 'paper',
                                             'statement_reference_keys': [], 'paper_id': ep['paper_id'],
                                             'source_paper_id': err.get('source_paper_id', err.get('paper_id')),
                                             'source_version': err.get('source_version', err.get('version')),
                                             'version': err.get('target_version'),
                                             'version_scope': 'target_version_if_known_independent_erratum_version_kept_separately'})
                pair['evidence_documents'] = docs
                pair['evidence_collection'] = {'cohort': 'provided_existing_candidates' if args.input else 'prior_round4_automatically_eligible_only',
                    'public_errata_search': 'bounded_official_pages_and_explicitly_recorded_public_searches_not_exhaustive',
                    'negative_result_is_not_global_silence': True}
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        assert path.stat().st_size <= 128 * 1024 * 1024
        with path.open('rb') as f:
            assert all(len(line) <= CAP for line in f)
        stats[cohort + '_pairs'] = len(rows)
        stats[cohort + '_input_bytes'] = path.stat().st_size
    save(OUT / 'source_index.json', list(sources.values()))
    save(OUT / 'preparation_receipt.json', {'counts': dict(stats), 'unique_papers': len(pids),
        'unique_source_paths': len({r.get('ref') or r.get('path') for r in sources.values()}),
        'metadata_available': sum(bool(r.get('source_available')) for r in metadata.values()),
        'metadata_total': len(metadata), 'exact_membership': 'provided_existing_bounded_candidates' if args.input else 'round4_97_version_plus_2_forward',
        'provided_input': provided_info,
        'source_inputs': {str(args.input.resolve()): provided_info['sha256']} if args.input else {p: hash_file(HERE / 'inputs' / p) for p in ('versions40.jsonl', 'forward96.full.jsonl')},
        'inputs': {c: {'path': str(OUT / 'inputs' / (c + '.jsonl')), 'sha256': sha((OUT / 'inputs' / (c + '.jsonl')).read_bytes())} for c in cohorts},
        'algorithmic_verdict_inputs': [], 'limits': {'per_file_bytes': CAP, 'per_jsonl_line_bytes': CAP, 'selected_input_bytes': 128*1024*1024}})
    resolver = default_resolver()
    if resolver:
        resolver.save_catalog()
    print(json.dumps(dict(stats), ensure_ascii=False))


if __name__ == '__main__':
    main()
