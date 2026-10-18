"""Bounded TeX-aware clause spans and literal equation-reference evidence.

No TeX execution, source IO, semantic equivalence, or per-paper exceptions.
"""
import re
from hashlib import sha256
from statement_text import _active_text, _MATH_BLOCK, _math_normalize, _paper_id, _version
from static_tex_context import group

MAX_CLAUSE = 6000
REF = re.compile(r'\\(?:eqref|[cC]ref|[rR]ef)\s*\{([^{}]+)\}')
CUE = re.compile(r'\b(?P<cue>if|suppose(?:\s+that)?|assume(?:\s+(?:also\s+)?that)?|assuming(?:\s+that)?|provided(?:\s+that)?)\s+', re.I)


def masks(text):
    """Coordinates stay in the original statement; notes remain separate evidence."""
    active = _active_text(text)
    chars = list(active)
    notes = []
    for m in re.finditer(r'\\(?:footnote|footnotetext)\*?(?:\s*\[[^\]]*\])?\s*', active):
        if any(a <= m.start() < b for a, b in notes):
            continue
        g = group(active, m.end())
        end = g[1] if g else len(active)
        notes.append((m.start(), end))
        chars[m.start():end] = ' ' * (end-m.start())
    semantic = ''.join(chars)
    for m in _MATH_BLOCK.finditer(semantic):
        chars[m.start():m.end()] = ' ' * (m.end()-m.start())
    # Protect punctuation and cues in macro arguments and prose parentheses.
    brace = paren = 0
    for i, ch in enumerate(semantic):
        escaped = i > 0 and semantic[i-1] == '\\'
        if not escaped:
            if ch == '{': brace += 1
            if ch == '(': paren += 1
        if brace or paren: chars[i] = ' '
        if not escaped:
            if ch == '}': brace = max(0, brace-1)
            if ch == ')': paren = max(0, paren-1)
    return active, semantic, ''.join(chars), notes


class Clause:
    def __init__(self, text, start, body_start, end):
        self.text, self.a, self.b, self.z = text, start, body_start, end

    def start(self): return self.a
    def end(self): return self.z
    def group(self, key=0):
        return self.text[self.b if key == 'body' else self.a:self.z]


def clauses(text):
    active, semantic, shadow, _ = masks(text)
    boundary = re.search(r'\bthen\b|\bwe\s+have\b|\\(?:Longrightarrow|implies)\b|⇒', shadow, re.I)
    if not boundary:
        return []
    limit = boundary.start()
    found = []
    for m in CUE.finditer(shadow[:limit]):
        body_start = m.end('cue')
        while body_start < limit and active[body_start].isspace(): body_start += 1
        if re.search(r'\bonly\s*$', shadow[:m.start()], re.I): continue
        if re.match(r'and\s+only\s+if\b', shadow[m.end():], re.I): continue
        # Commas inside formulas, braces, notes and parentheticals were masked.
        stop = re.search(r'[.;](?=\s|$)|,(?=\s*(?:then|where|if|assum|suppos|provided)\b)|\bthen\b', shadow[m.end():limit], re.I)
        end = m.end()+stop.start() if stop else limit
        next_cue = CUE.search(shadow, m.end(), end)
        if next_cue: end = next_cue.start()
        # Remove a final separator before Then, without dropping a predicate.
        while end > m.end() and active[end-1] in ' \t\r\n,;': end -= 1
        if end-m.start() > MAX_CLAUSE: continue  # Never promote a truncated clause.
        fragment = semantic[body_start:end]
        unmatched = _MATH_BLOCK.sub('', fragment)
        if re.search(r'(?<!\\)\$|\\[\[(]', unmatched): continue
        if any(c.start() <= m.start() < c.end() for c in found): continue
        found.append(Clause(active, m.start(), body_start, end))
    return found


def body_key(text):
    # Same text after note removal and harmless whitespace only; retain case.
    return re.sub(r'\s+', ' ', masks(text)[1]).strip().rstrip('.,;')


def subject(text):
    value = masks(text)[1]
    _, _, shadow, _ = masks(value)
    copula = re.search(r'\b(?:is|are|be|has|have|satisfies|satisfy|lies|lie|holds|hold)\b', shadow, re.I)
    if not copula: return None
    raw = value[:copula.start()].strip()
    raw = re.sub(r'^the\s+(?:following\s+)?', '', raw, flags=re.I)
    raw = re.sub(r'\s+in\s+\\(?:eqref|[cC]ref|[rR]ef)\s*\{[^{}]+\}\s*$', '', raw)
    return re.sub(r'\s+', '', raw)


def comparison(old, new):
    """Return complete aligned clauses; a changed old predicate is not an addition."""
    if re.search(r'\\item\b|\\begin\s*\{(?:enumerate|cases)\}', old + new): return []
    aa, bb = clauses(old), clauses(new)
    rows = []
    for b in bb:
        key = subject(b.group('body'))
        matches = [a for a in aa if key and subject(a.group('body')) == key]
        if len(matches) != 1: continue
        a = matches[0]
        same = body_key(a.group('body')) == body_key(b.group('body'))
        rows.append({'status': 'same_main_clause' if same else 'changed_existing_clause',
                     'old_span': [a.start(), a.end()], 'new_span': [b.start(), b.end()],
                     'old_quote': a.group(), 'new_quote': b.group(),
                     'subject_key': key, 'logical_direction': 'unknown',
                     'basis': 'unique_equal_case_sensitive_subject_before_copula'})
    return rows


def unmatched_clauses(old, new, matched):
    """Expose unpaired hypotheses, without inventing a shared subject or direction."""
    output = {}
    for side, text in [('old', old), ('new', new)]:
        covered = {tuple(row[side+'_span']) for row in matched}
        output[side] = [{'quote': c.group(), 'span': [c.start(),c.end()]} for c in clauses(text)
                        if (c.start(),c.end()) not in covered]
    old_keys=[body_key(c['quote']) for c in output['old']]
    new_keys=[body_key(c['quote']) for c in output['new']]
    shared={key for key in old_keys if old_keys.count(key)==new_keys.count(key)==1}
    for side in ('old','new'):
        output[side]=[c for c in output[side] if body_key(c['quote']) not in shared]
    if [body_key(c['quote']) for c in output['old']] == [body_key(c['quote']) for c in output['new']]: return {}
    return {**output, 'status': 'clause_correspondence_unresolved', 'added_premise_certified': False}


def reference_evidence(pair, old, new, oldtext, newtext, reverse=False):
    """Resolve unique equation labels in bounded, supplied same-endpoint documents.

    Includes both inline labelled displays and ref calls. Missing/duplicate labels
    stay explicit; matching label names alone never establish unchanged content.
    """
    outputs = {}
    for name, side, statement in [('old', old, oldtext), ('new', new, newtext)]:
        label_keys = set(REF.findall(statement))
        for block in _MATH_BLOCK.finditer(_active_text(statement)):
            label_keys.update(re.findall(r'\\label\s*\{([^{}]+)\}', block.group()))
        docs = []
        for doc in pair.get('evidence_documents', [])[:12]:
            if not isinstance(doc, dict): continue
            original_side = ('new' if name == 'old' else 'old') if reverse else name
            if (doc.get('source_type') != 'full_tex' or doc.get('side') != original_side
                    or _paper_id(doc) != _paper_id(side) or _version(doc) != _version(side)
                    or not _paper_id(side) or _version(side) is None
                    or doc.get('text_truncated') or not isinstance(doc.get('text'), str)
                    or len(doc['text']) > 1000000): continue
            text = doc['text']
            if statement not in text: continue
            digest = sha256(text.encode('utf-8')).hexdigest()
            if not any(d[0] == digest for d in docs): docs.append((digest, text, doc.get('doc_id')))
        entries = {}
        for label in sorted(label_keys)[:32]:
            targets = []
            for digest, source, doc_id in (docs if len(docs) == 1 else []):
                active = _active_text(source)
                hits = list(re.finditer(r'\\label\s*\{' + re.escape(label) + r'\}', active))
                if len(hits) != 1: continue
                hit = hits[0]
                document_end = active.find(r'\end{document}')
                if document_end >= 0 and hit.start() > document_end: continue
                depth = 0
                for conditional in re.finditer(r'\\(?:if[A-Za-z]*|fi)(?![A-Za-z])', active[:hit.start()]):
                    depth = max(0, depth-1) if conditional.group() == r'\fi' else depth+1
                if depth: continue
                blocks = [b for b in _MATH_BLOCK.finditer(active) if b.start() <= hit.start() < b.end()]
                if len(blocks) != 1 or blocks[0].end()-blocks[0].start() > 12000: continue
                block = blocks[0]
                if REF.search(block.group()): continue  # No recursive expansion.
                targets.append({'quote': source[block.start():block.end()],
                    'normalized': _math_normalize(block.group()), 'source_span': [block.start(), block.end()],
                    'start_line': source.count('\n', 0, block.start())+1,
                    'end_line': source.count('\n', 0, block.end())+1,
                    'doc_id': doc_id, 'source_text_sha256': digest,
                    'coordinate_basis': 'provided_evidence_document_text_not_canonical_viewer_lines'})
            entries[label] = {'status': 'unique_equation_target' if len(targets) == 1 else 'unresolved_or_ambiguous',
                              'target': targets[0] if len(targets) == 1 else None}
        outputs[name] = entries
    comparisons = []
    for key in sorted(outputs['old'].keys() & outputs['new'].keys()):
        a, b = outputs['old'][key]['target'], outputs['new'][key]['target']
        if a and b:
            comparisons.append({'label': key, 'status': 'same_literal_equation' if a['normalized'] == b['normalized'] else 'changed_equation',
                                'old': a, 'new': b, 'mathematical_equivalence_certified': False})
    return {'endpoints': outputs, 'comparisons': comparisons,
            'scope': 'supplied_version_bound_complete_text_unique_math_labels_only_no_recursive_or_semantic_expansion'}


def note_changes(old, new):
    aa, _, _, an = masks(old)
    bb, _, _, bn = masks(new)
    before = [aa[a:b] for a, b in an]
    after = [bb[a:b] for a, b in bn]
    if before == after: return []
    return [{'old_notes': before, 'new_notes': after, 'status': 'note_text_changed_scope_unverified',
             'note': '脚注内容单独保留；未当作主句前提，也未作为纯改写删除。'}]
