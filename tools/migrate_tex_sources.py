"""Bounded, offline source migration for the already evaluated 476 pairs.

Plan -> verify reports -> archive exact legacy evidence -> remove listed files.
The detector is never executed and its result JSONL is never rewritten.
"""
from collections import Counter, defaultdict
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile
from urllib.parse import urlsplit

from tex_sources import CorpusStore, SourceResolver, bind_cached_version, digest, normalise_newlines, source_key

HERE = Path(os.environ.get('THEOREM_ALGORITHM_WORKSPACE', Path(__file__).resolve().parent)).resolve()
ROOT = HERE.parent
ROUND = HERE / 'formula_scope_round_20261018'
RUN = ROUND / 'final_0_4_4'
OUT = HERE / 'source_migration_20261018'
CONFIG = HERE / 'source_config.local.json'


def save(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(destination)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def results_files():
    return sorted(RUN.glob('*/results.jsonl'))


def native_binding(path, version, corpus):
    return bind_cached_version(path, version, corpus)

def plan():
    if (OUT / 'cleanup_receipt.json').exists():
        raise RuntimeError('Migration already cleaned; do not rediscover from removed files')
    OUT.mkdir(exist_ok=True)
    if not CONFIG.exists():
        raise ValueError('Provide source_config.local.json explicitly; corpus paths and indexes are never guessed or rebuilt.')
    config = read(CONFIG)
    store = CorpusStore(**config['corpus'])
    groups = defaultdict(list)
    hashes = {}
    for file in results_files():
        hashes[str(file)] = sha(file)
        for pair in rows(file):
            for side in ('old', 'new'):
                ep = pair[side]
                groups[ep['paper_id']].append((pair['pair_id'], side, ep))
    sources, aliases, bindings, canonical, deletions, endpoints = {}, {}, {}, {}, {}, []
    coverage, matches = Counter(), Counter()
    attempted = set()
    from prepare_recheck99 import locate

    def add_cached(path, ep):
        path = path.resolve()
        if not path.is_file() or path.suffix != '.tex':
            return None
        data = path.read_bytes()
        if len(data) > 8 * 1024**2:
            raise ValueError('Cached source exceeds bound')
        ref = 'cache-' + digest(data)[:24]
        sources.setdefault(ref, {'backend': 'historical_cache', 'path': str(path),
            'sha256': digest(data), 'bytes': len(data), 'ref': ref,
            'paper_id': ep['paper_id'], 'version': ep.get('version'),
            'full_tex_available': True, 'source_scope': 'prior_cached_source_exact_bytes'})
        aliases[str(path)] = ref
        return ref

    for number, (pid, items) in enumerate(sorted(groups.items()), 1):
        corpus = store.get(pid)
        for pair_id, side, ep in items:
            access = ep.get('source', {}).get('source_access', {})
            path_value = access.get('path') or ep.get('source', {}).get('path')
            path = Path(path_value).resolve() if path_value else None
            ref = add_cached(path, ep) if path else None
            key = source_key(ep)
            attempt_key = (key, str(path))
            if corpus is not None and ep.get('version') is not None and key not in bindings and path and path.is_file() and attempt_key not in attempted:
                attempted.add(attempt_key)
                binding = native_binding(path, ep['version'], corpus)
                if binding:
                    bindings[key] = binding
            use_corpus = corpus is not None and (ep.get('version') is None or key in bindings)
            if use_corpus:
                cref = 'corpus-' + digest((key + corpus['text_sha256']).encode())[:24]
                sources.setdefault(cref, {'backend': 'corpus', 'paper_id': pid, 'version': ep.get('version'),
                    'sha256': corpus['text_sha256'], 'bytes': len(corpus['text'].encode()),
                    'corpus_locator': corpus['corpus_locator'], 'ref': cref, 'path': None,
                    'full_tex_available': True, 'source_scope': 'original_corpus_text_read_in_place'})
                canonical[key] = cref
                if path and path.is_file():
                    ptext = path.read_bytes()
                    same_bytes = digest(ptext) == corpus['text_sha256']
                    # Unknown-version snapshots are reusable for navigation, but
                    # old bytes are removable only when exact or version-bound.
                    if same_bytes or key in bindings:
                        deletions[str(path)] = {'path': str(path), 'sha256': digest(ptext),
                            'bytes': len(ptext), 'canonical_ref': cref,
                            'recovery': 'original_corpus_exact_bytes' if same_bytes else 'retired_evidence_archive',
                            'reason': 'exact_full_corpus_copy' if same_bytes else 'expanded_copy_of_native_bound_corpus_version'}
                        if same_bytes and ref:
                            sources[ref] = {**sources[cref], 'ref': ref}
                        if key in bindings:
                            main = Path(bindings[key]['original_member'])
                            if main.is_file():
                                mref = add_cached(main, ep)
                                deletions[str(main)] = {'path': str(main), 'sha256': sha(main),
                                    'bytes': main.stat().st_size, 'canonical_ref': cref,
                                    'recovery': 'retired_evidence_archive', 'reason': 'unpacked_main_file_already_in_original_corpus'}
            endpoints.append({'pair_id': pair_id, 'side': side, 'key': key, 'old_path': str(path) if path else None,
                              'canonical_ref': canonical.get(key), 'legacy_ref': ref})
            coverage['corpus' if use_corpus else 'historical_cache' if ref else 'missing'] += 1
            if use_corpus or ref:
                text = corpus['text'] if use_corpus else Path(sources[ref]['path']).read_bytes().decode('utf-8', errors='replace')
                _, _, method = locate(text.replace('\r\n', '\n'), ep['text'].replace('\r\n', '\n'))
                matches[method] += 1
        if number % 20 == 0:
            print(json.dumps({'original_corpus_papers_read': number, 'target_papers': len(groups)}, ensure_ascii=False), flush=True)
    catalog = {'schema': 'original-corpus-first-v1', 'version_bindings': bindings,
        'canonical_by_key': canonical, 'sources': sources, 'legacy_paths': aliases,
        'endpoints': endpoints, 'frozen_results_sha256': hashes,
        'scope': '952 endpoints of already evaluated476; no full-corpus query or index rebuild'}
    save(OUT / 'source_catalog.json', catalog)
    save(OUT / 'cleanup_plan.json', {'status': 'PLANNED_NOT_REMOVED', 'files': list(deletions.values()),
        'bytes': sum(d['bytes'] for d in deletions.values()), 'allowed_root': str(ROOT.resolve()),
        'corpus_root_read_only': str(store.body_root), 'algorithm_results_sha256': hashes})
    summary = {'status': 'PLANNED', 'unique_papers_read': len(groups), 'endpoints': len(endpoints),
        'endpoint_backends': dict(coverage), 'source_matches': dict(matches),
        'native_full_source_version_bindings': len(bindings), 'removable_files': len(deletions),
        'removable_uncompressed_bytes': sum(d['bytes'] for d in deletions.values()),
        'result_files_unchanged': all(sha(p) == h for p, h in hashes.items()),
        'full_corpus_rerun': False, 'network_fetches': 0}
    save(OUT / 'planning_verification.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def cleanup():
    if (OUT / 'cleanup_receipt.json').exists():
        print(json.dumps({'status': 'ALREADY_CLEANED', 'receipt': str(OUT / 'cleanup_receipt.json')}))
        return
    plan = read(OUT / 'cleanup_plan.json')
    verification = read(OUT / 'migration_verification.json')
    if verification['status'] != 'PASSED':
        raise ValueError('Reports and sources must pass verification before cleanup')
    allowed = Path(plan['allowed_root']).resolve()
    forbidden = Path(plan['corpus_root_read_only']).resolve()
    for entry in plan['files']:
        path = Path(entry['path']).resolve()
        if not path.is_relative_to(allowed) or path == allowed or path.is_relative_to(forbidden):
            raise ValueError('Deletion target outside verified candidate cache workspace')
        if path.exists() and (not path.is_file() or sha(path) != entry['sha256']):
            raise ValueError('Deletion target changed since planning')
    archive_path = OUT / 'retired_evidence.zip'
    catalog = read(OUT / 'source_catalog.json')
    archived = []
    with zipfile.ZipFile(archive_path, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        present = set(archive.namelist())
        for entry in plan['files']:
            if entry['recovery'] != 'retired_evidence_archive':
                continue
            member = 'tex/' + entry['sha256'] + '.tex'
            if member not in present:
                archive.writestr(member, Path(entry['path']).read_bytes())
                present.add(member)
            if digest(archive.read(member)) != entry['sha256']:
                raise ValueError('Retired evidence archive did not round-trip')
            ref = catalog['legacy_paths'][entry['path']]
            catalog['sources'][ref].update(backend='legacy_archive', path=None,
                archive_path=str(archive_path), archive_member=member)
            archived.append(member)
    # Commit readable recovery registry before removing any file. No recursive
    # filesystem delete and no computed path outside the explicit allow-list.
    save(OUT / 'source_catalog.json', catalog)
    resolver = SourceResolver(read(CONFIG))
    removed = []
    for entry in plan['files']:
        ref = catalog['legacy_paths'][entry['path']]
        recovered = resolver.read_bytes(catalog['sources'][ref])
        if digest(recovered) != entry['sha256']:
            raise ValueError('Evidence recovery hash failed before deletion')
        if Path(entry['path']).exists():
            Path(entry['path']).unlink()
        removed.append(entry)
    for path, expected in plan['algorithm_results_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen algorithm result changed')
    for entry in removed:
        assert not Path(entry['path']).exists()
        ref = catalog['legacy_paths'][entry['path']]
        assert digest(resolver.read_bytes(catalog['sources'][ref])) == entry['sha256']
    save(OUT / 'cleanup_receipt.json', {'status': 'COMPLETED',
        'removed_files': removed, 'removed_bytes': sum(d['bytes'] for d in removed),
        'retired_archive_bytes': archive_path.stat().st_size,
        'archived_unique_tex_members': len(set(archived)),
        'recovery_hashes_checked_before_removal': len(removed),
        'recovery_hashes_checked_after_removal': len(removed),
        'frozen_algorithm_results_unchanged': True, 'recursive_deletion_used': False,
        'corpus_modified': False, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'status': 'CLEANED', 'files': len(removed),
        'bytes_removed': sum(d['bytes'] for d in removed), 'archive_bytes': archive_path.stat().st_size}))


def active_reports():
    """Reader products, not frozen code/results or recorded reader snapshots."""
    old = HERE / 'recheck_99_20261018'
    files = set(HERE.glob('*.md'))
    for root in (ROUND, old):
        files.update(root.glob('*.md'))
        files.update(root.glob('*.txt'))
    for root in (ROUND / 'categories', ROUND / 'reader_cards', old / 'quick_cards', old / 'model_cards'):
        files.update(root.glob('*.md'))
        files.update(root.glob('*.json'))
    files.update(ROUND / name for name in ('reader_packet_index.json', 'user_views_manifest.json'))
    files.update(old / name for name in ('model_packet.json', 'model_packet_index.json', 'user_views_manifest.json'))
    for root in (RUN, old / 'final_0_3_1'):
        files.update(root.glob('*/*.md'))
    return sorted(p for p in files if p.is_file())


def backup_reports():
    manifest = OUT / 'reports_before_migration.json'
    if manifest.exists():
        raise ValueError('Reader originals already preserved; do not overwrite')
    records = []
    with zipfile.ZipFile(OUT / 'retired_evidence.zip', 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        present = set(archive.namelist())
        for path in active_reports():
            h = sha(path)
            member = 'reports/' + h + path.suffix
            if member not in present:
                archive.writestr(member, path.read_bytes())
                present.add(member)
            assert digest(archive.read(member)) == h
            records.append({'path': str(path), 'sha256': h, 'bytes': path.stat().st_size, 'archive_member': member})
    save(manifest, {'status': 'PRESERVED', 'files': records,
                    'archive': str(OUT / 'retired_evidence.zip'),
                    'frozen_reader_feedback_and_snapshots': 'left byte-for-byte unchanged'})
    print(json.dumps({'report_originals_preserved': len(records)}))


def relink_reports():
    catalog = read(OUT / 'source_catalog.json')
    resolver = SourceResolver(read(CONFIG))
    before = read(OUT / 'reports_before_migration.json')
    backed = {r['path'] for r in before['files']}
    aliases = {p.replace('\\', '/').casefold(): ref for p, ref in catalog['legacy_paths'].items()}
    edits = []
    pattern = re.compile(r'\]\((?:<)?([A-Za-z]:[\\/][^>\n]+?)(?:>)?\)')
    for path in active_reports():
        if path.suffix != '.md':
            continue
        text = path.read_text(encoding='utf-8')
        count = [0]
        def replace(match):
            target = match[1]
            line_match = re.search(r':(\d+)$', target)
            line = int(line_match[1]) if line_match else None
            original = target[:line_match.start()] if line_match else target
            ref = aliases.get(original.replace('\\', '/').casefold())
            if ref is None:
                return match[0]
            count[0] += 1
            # Non-regenerated historical views retain their exact old bytes and
            # old line coordinates; the current reader packs use canonical text.
            return '](<' + resolver.url(ref, line) + '>)'
        revised = pattern.sub(replace, text)
        if revised != text:
            assert str(path) in backed
            path.write_text(revised, encoding='utf-8')
            edits.append({'path': str(path), 'links_updated': count[0], 'sha256': sha(path)})
    save(OUT / 'report_link_updates.json', {'files': edits,
        'new_current_reader_views': str(ROUND / 'reader_packet_index.json'),
        'historical_snapshot_policy': 'frozen snapshots retain original hashes; their original source paths resolve through source_catalog.legacy_paths'})
    print(json.dumps({'additional_reports_relinked': len(edits), 'links': sum(r['links_updated'] for r in edits)}))


def verify():
    from prepare_recheck99 import locate
    resolver = SourceResolver(read(CONFIG))
    plan = read(OUT / 'cleanup_plan.json')
    for path, expected in plan['algorithm_results_sha256'].items():
        assert sha(path) == expected
    for name in ('strength_signals.py', 'formula_scope.py', 'static_tex_context.py'):
        assert sha(HERE / name) == sha(ROUND / 'code' / name)
    grouped = defaultdict(list)
    for path in results_files():
        for row in rows(path):
            for side in ('old', 'new'):
                grouped[row[side]['paper_id']].append((row['pair_id'], side, row[side]))
    counts, matches = Counter(), Counter()
    unmatched = []
    for number, (pid, items) in enumerate(sorted(grouped.items()), 1):
        for pair, side, ep in items:
            access = resolver.resolve(ep)
            counts[access['backend']] += 1
            if not access.get('full_tex_available'):
                continue
            data = resolver.read_bytes(access)
            assert digest(data) == access['sha256']
            text = data.decode('utf-8', errors='replace').replace('\r\n', '\n')
            _, _, method = locate(text, ep['text'].replace('\r\n', '\n'))
            matches[method] += 1
            if method == 'not_located_in_full_source':
                unmatched.append({'pair_id': pair, 'side': side,
                    'previous_full_tex_available': ep.get('source', {}).get('source_access', {}).get('full_tex_available'),
                    'backend': access['backend']})
        if number % 50 == 0:
            print(json.dumps({'verified_original_corpus_papers': number, 'target': len(grouped)}), flush=True)
    # Existing source coverage must never regress. Newly filled unknown-version
    # sources may be readable without an exact statement span; show that openly.
    assert all(not r['previous_full_tex_available'] for r in unmatched)
    index = read(ROUND / 'reader_packet_index.json')
    for card in index['cards']:
        assert sha(card['quick_json']) == card['quick_sha256']
        assert sha(card['details_json']) == card['details_sha256']
        quick = read(card['quick_json'])
        for side in ('old', 'new'):
            a = quick[side]['full_tex']
            if a.get('full_tex_available') and a.get('statement_start_line'):
                text = resolver.read(a).replace('\r\n', '\n')
                assert a['statement_start_line'] <= len(text.splitlines())
        for anchor in quick.get('referenced_source_anchors', []):
            text = resolver.read(anchor['source_access'])
            assert re.search(r'\\label\s*\{\s*' + re.escape(anchor['label']) + r'\s*\}', text.splitlines()[anchor['line']-1])
    refs = resolver.catalog['sources']
    viewer_links = 0
    retired_targets = {p['path'].replace('\\', '/').casefold() for p in plan['files']}
    for path in active_reports():
        if path.suffix != '.md':
            continue
        text = path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(<?([^>\n]+?)>?\)', text):
            if target.startswith(resolver.base_url + '/tex/'):
                ref = urlsplit(target).path.removeprefix('/tex/')
                assert ref in refs, (path, ref)
                viewer_links += 1
            raw_target = re.sub(r':\d+$', '', target).replace('\\', '/').casefold()
            assert raw_target not in retired_targets, (path, target)
    before = read(OUT / 'reports_before_migration.json')
    with zipfile.ZipFile(OUT / 'retired_evidence.zip') as archive:
        for entry in before['files']:
            assert digest(archive.read(entry['archive_member'])) == entry['sha256']
    for entry in plan['files']:
        ref = resolver.catalog['legacy_paths'][entry['path']]
        assert digest(resolver.read_bytes(refs[ref])) == entry['sha256']
    result = {'status': 'PASSED', 'algorithm_result_files_unchanged': len(plan['algorithm_results_sha256']),
        'algorithm_core_three_files_unchanged': True, 'target_papers': len(grouped),
        'endpoints': sum(counts.values()), 'endpoint_backends': dict(counts), 'source_matches': dict(matches),
        'newly_filled_sources_without_statement_span': unmatched, 'reader_cards': len(index['cards']),
        'active_viewer_links_verified': viewer_links, 'active_links_to_retired_files': 0,
        'original_reports_recoverable': len(before['files']), 'retiring_sources_recoverable': len(plan['files']),
        'full_corpus_rerun': False, 'network_fetches': 0,
        'cleanup_already_completed': (OUT / 'cleanup_receipt.json').exists(),
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save(OUT / 'migration_verification.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'backup-reports', 'relink-reports', 'verify', 'cleanup'])
    args = parser.parse_args()
    {'plan': plan, 'cleanup': cleanup, 'backup-reports': backup_reports,
     'relink-reports': relink_reports, 'verify': verify}[args.command]()


if __name__ == '__main__':
    main()
