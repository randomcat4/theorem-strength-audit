"""Conservative statement-number hints for standard static newtheorem counters.

Not a TeX interpreter. Unsupported counters, conditional body, or format
overrides produce unknown, rather than guessed printed identities.
"""
import re


def mask_comments(text):
    return re.sub(r'(?<!\\)%[^\n]*', lambda m: ' ' * len(m.group()), text)


def statement_reference(text, body_start):
    raw = mask_comments(text)
    document = raw.find(r'\begin{document}')
    unknown = {'status': 'unknown_unsupported_static_numbering', 'keys': [], 'tex_execution_performed': False}
    if document < 0 or body_start < document or raw.count(r'\begin{document}') != 1:
        return unknown
    definitions = {}
    pattern = r'\\newtheorem(\*)?\s*\{([^{}]+)\}\s*(?:\[([^\]]+)\])?\s*\{([^{}]+)\}\s*(?:\[([^\]]+)\])?'
    for m in re.finditer(pattern, raw[:document]):
        star, env, shared, title, within = m.groups()
        if env in definitions or '\\' in title or (shared and within):
            return unknown
        definitions[env] = {'numbered': not star, 'shared': shared, 'title': title, 'within': within}
    if not definitions:
        return unknown

    def root(env):
        visited = set()
        while env in definitions and definitions[env]['shared']:
            if env in visited:
                return None
            visited.add(env)
            env = definitions[env]['shared']
        return env if env in definitions else None

    roots = {root(env) for env in definitions}
    if None in roots:
        return unknown
    opening_at_start = re.match(r'\\begin\{[^{}]+\}', raw[body_start:])
    event_limit = body_start + (opening_at_start.end() if opening_at_start else 1)
    prior_environments = [m.group(1) for m in re.finditer(r'\\begin\{([^{}]+)\}', raw[document:event_limit])
                          if m.group(1) in definitions]
    if not prior_environments:
        return unknown
    target_root = root(prior_environments[-1])
    relevant = {target_root, 'section', 'subsection'}
    prior = raw[:body_start]
    for counter in relevant:
        if re.search(r'\\(?:setcounter|addtocounter|stepcounter|refstepcounter)\s*\{\s*' + re.escape(counter) + r'\s*\}', prior):
            return unknown
        if re.search(r'\\(?:renewcommand|def)\s*\{?\\the' + re.escape(counter) + r'\b', prior):
            return unknown
    active = raw[document:body_start]
    if re.search(r'\\(?:appendix|includeonly|if[a-zA-Z]+|else|fi)\b', active):
        return unknown
    for item in definitions.values():
        if item['within'] not in (None, 'section', 'subsection'):
            return unknown
    counters = {counter: 0 for counter in roots}
    section = subsection = 0
    # Body start can point at the first character after the begin environment.
    events = list(re.finditer(r'\\(subsection|section)(\*)?\s*\{|\\begin\{([^{}]+)\}', raw[document:event_limit]))
    last = None
    for event in events:
        kind, star, env = event.groups()
        if kind:
            if star:
                continue
            if kind == 'section':
                section += 1
                subsection = 0
            else:
                subsection += 1
            for counter in roots:
                within = definitions[counter]['within']
                if within == kind or (kind == 'section' and within == 'subsection'):
                    counters[counter] = 0
            continue
        if env not in definitions:
            continue
        definition = definitions[env]
        if not definition['numbered']:
            last = None
            continue
        counter = root(env)
        counters[counter] += 1
        within = definitions[counter]['within']
        number = str(counters[counter])
        if within == 'section':
            number = f'{section}.{number}'
        elif within == 'subsection':
            number = f'{section}.{subsection}.{number}'
        last = {'environment': env, 'title': definition['title'], 'number': number,
                'begin_offset': document + event.start(), 'key': definition['title'] + ' ' + number}
    if last is None:
        return unknown
    # A preceding theorem is not evidence for text outside its environment.
    close = re.search(r'\\end\{' + re.escape(last['environment']) + r'\}', raw[last['begin_offset']:])
    if close is None or last['begin_offset'] + close.start() < body_start:
        return unknown
    return {'status': 'supported_standard_newtheorem_static_hint', 'keys': [last['key']], **last,
            'tex_execution_performed': False, 'basis': 'newtheorem_shared_counters_section_events_no_supported_counter_overrides'}
