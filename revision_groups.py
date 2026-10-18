"""Group literal repeated endpoints within the same version transition.

All pair rows and their evidence stay present. This is a reading unit, not a
certified mathematical revision event or an alignment-gate override.
"""
import re
from hashlib import sha256
from statement_text import _active_text, _paper_id, _version


def literal_key(text):
    text = _active_text(text)
    text = re.sub(r'\\label\s*\{[^{}]*\}', '', text)
    # Ignore only numbering of the same display environment, not its contents.
    text = re.sub(r'(\\(?:begin|end)\s*\{(?:equation|align|gather))\*(\})', r'\1\2', text)
    # Keep word boundaries: "a part" and "apart" are not the same text.
    return '\0'.join(re.findall(r'\\[A-Za-z]+|[A-Za-z]+|[0-9]+|[^\s]', text))


def group_restatements(records):
    groups = {}
    for r in records:
        a, b = r['old'], r['new']
        if any(s.get('display_text_truncated') or s.get('input_text_length', 0) > 24000
               or s.get('source', {}).get('input_text_truncated') for s in (a, b)): continue
        identity = [_paper_id(a), _version(a), _paper_id(b), _version(b)]
        if not all(identity) or identity[0] != identity[2]: continue
        texts = [literal_key(s.get('text', '')) for s in (a, b)]
        if not all(texts): continue
        key = sha256(repr(identity + texts).encode('utf-8')).hexdigest()[:24]
        groups.setdefault(key, []).append(r)
    for key, members in groups.items():
        ids = sorted(r['pair_id'] for r in members)
        for r in members:
            r['algorithm']['reading_group'] = {
                'group_id': 'reading_' + key, 'representative_pair_id': ids[0],
                'member_pair_ids': ids, 'pair_count': len(ids),
                'repeated_text': len(ids) > 1,
                'basis': 'same_paper_versions_and_literal_endpoints_ignoring_labels_whitespace_display_numbering',
                'certified_independent_events': None, 'raw_rows_preserved': True}
