"""Bounded static numbered-list references, with content-based correspondence.

Only explicit ranges near section references are supported. Unknown numbering,
multiple targets and partial text remain unresolved; no TeX macros are executed.
"""
import re
from difflib import SequenceMatcher
from hashlib import sha256
from statement_text import _active_text, _MATH_BLOCK, _math_normalize, _paper_id, _version
from static_tex_context import group
from premise_clauses import REF


def plain_marker(text):
    return re.sub(r'\\[A-Za-z]+\*?', '', text).translate(str.maketrans('', '', '{}')).strip()


def item_key(text):
    tokens=[];last=0
    for m in _MATH_BLOCK.finditer(_active_text(text)):
        tokens.extend(re.findall(r'[A-Za-z]+|[^\s]',text[last:m.start()]))
        tokens.append(('math',_math_normalize(m.group())))
        last=m.end()
    tokens.extend(re.findall(r'[A-Za-z]+|[^\s]',text[last:]))
    return repr(tokens)


def text_match_score(left, right, left_key, right_key):
    """Candidate ordering only. Formula anchors can survive added prose comments."""
    if max(len(left_key),len(right_key))>3000:return 0.0,'over_bound'
    score=SequenceMatcher(None,left_key,right_key,autojunk=True).ratio()
    basis='whole_item_text'
    aa=list(_MATH_BLOCK.finditer(left));bb=list(_MATH_BLOCK.finditer(right))
    if aa and bb:
        a,b=_math_normalize(aa[0].group()),_math_normalize(bb[0].group())
        if 32<=min(len(a),len(b)) and max(len(a),len(b))<=2000:
            formula_score=SequenceMatcher(None,a,b,autojunk=True).ratio()
            if formula_score>score:score,basis=formula_score,'first_formula_text_not_semantic_identity'
    return score,basis


def numbered_items(source, start, end):
    text=_active_text(source)
    region=text[start:end]
    if re.search(r'\\(?:setcounter|addtocounter|renewcommand|def)\b',region): return None
    events=re.finditer(r'\\(?:(begin|end)\s*\{(enumerate|itemize)\}|(item)(?![A-Za-z]))',region)
    stack=[];current=None;items={}
    def finish(stop):
        nonlocal current
        if current:
            number, body=current
            if number in items or stop-body>12000: return False
            quote=source[body:stop].strip()
            items[number]={'number':number,'quote':quote,'source_span':[body,stop],
                'start_line':source.count('\n',0,body)+1,'end_line':source.count('\n',0,stop)+1}
            current=None
        return True
    for m in events:
        kind, env, item=m.groups();absolute=start+m.start();following=start+m.end()
        if kind=='begin':
            cursor=following
            while cursor<end and text[cursor].isspace():cursor+=1
            option=''
            if text[cursor:cursor+1]=='[':
                close=text.find(']',cursor,min(end,cursor+300))
                if close<0:return None
                option=text[cursor+1:close]
            supported=not option or (env=='enumerate' and r'\arabic*' in option and not re.search(r'\b(?:start|resume|series|ref)\b',option))
            stack.append({'env':env,'counter':0,'supported':supported})
        elif kind=='end':
            if not stack or stack[-1]['env']!=env:return None
            if len(stack)==1 and not finish(absolute):return None
            stack.pop()
        elif item and len(stack)==1:
            if not stack[-1]['supported']:return None
            if not finish(absolute):return None
            cursor=following
            while cursor<end and text[cursor].isspace():cursor+=1
            if text[cursor:cursor+1]=='[':
                close=text.find(']',cursor,min(end,cursor+300))
                if close<0:return None
                marker=plain_marker(text[cursor+1:close])
                number=re.fullmatch(r'\(?([1-9]\d?)\)?[.)]?',marker)
                if not number:return None
                value=int(number.group(1));cursor=close+1
            elif stack[-1]['env']=='enumerate':
                stack[-1]['counter']+=1;value=stack[-1]['counter']
            else:return None
            current=(value,cursor)
            if len(items)>=64:return None
    return None if stack else items


def endpoint_ranges(pair, side_name, side, statement, reverse=False):
    output=[]
    for ref in list(REF.finditer(statement))[:32]:
        prefix=plain_marker(statement[max(0,ref.start()-300):ref.start()])
        select=re.search(r'\b(?:conditions|assumptions|hypotheses)\s*\(([1-9]\d?)\)\s*(?:--?|–|to)\s*\(([1-9]\d?)\)[^.\n]{0,100}$',prefix,re.I)
        if not select:continue
        first,last=map(int,select.groups())
        if first>last or last-first>32:continue
        label=ref.group(1)
        result={'label':label,'selected_numbers':list(range(first,last+1)), 'status':'unresolved_numbered_list',
                'coordinate_basis':'provided_evidence_document_text_not_canonical_viewer_lines'}
        docs={}
        expected=('new' if side_name=='old' else 'old') if reverse else side_name
        for d in pair.get('evidence_documents',[])[:12]:
            if (not isinstance(d,dict) or d.get('source_type')!='full_tex' or d.get('side')!=expected
                    or not _paper_id(side) or _version(side) is None
                    or _paper_id(d)!=_paper_id(side) or _version(d)!=_version(side)
                    or d.get('text_truncated') or not isinstance(d.get('text'),str)
                    or len(d['text'])>1000000 or statement not in d['text']):continue
            docs[sha256(d['text'].encode()).hexdigest()]=d
        if len(docs)!=1:output.append(result);continue
        digest,doc=next(iter(docs.items()));source=doc['text'];active=_active_text(source)
        labels=list(re.finditer(r'\\label\s*\{'+re.escape(label)+r'\}',active))
        if len(labels)!=1:output.append(result);continue
        hit=labels[0];headers=list(re.finditer(r'\\(?:section|subsection|subsubsection)\*?\s*',active[:hit.start()]))
        before=active[:hit.start()]
        depth=0
        for token in re.finditer(r'\\(?:if[A-Za-z]*|fi)(?![A-Za-z])',before):
            depth=max(0,depth-1) if token.group()==r'\fi' else depth+1
        if depth or r'\end{document}' in before:output.append(result);continue
        heading=headers[-1] if headers else None
        g=group(active,heading.end()) if heading else None
        # The label must actually belong to this section header.
        if not g or active[g[1]:hit.start()].strip():output.append(result);continue
        boundary=re.search(r'\\(?:section|subsection|subsubsection)\*?\s*\{|\\end\s*\{document\}',active[hit.end():])
        stop=hit.end()+boundary.start() if boundary else len(active)
        if stop-hit.end()>100000 or re.search(r'\\if[A-Za-z]*(?![A-Za-z])',active[hit.end():stop]):output.append(result);continue
        items=numbered_items(source,hit.end(),stop)
        if items is None or any(n not in items for n in result['selected_numbers']):output.append(result);continue
        result.update(status='resolved_static_numbered_list',items=items,doc_id=doc.get('doc_id'),source_text_sha256=digest,
                      scope='explicit_static_markers_not_executed_tex_or_semantic_assumptions')
        output.append(result)
    return output


def compare_condition_references(pair, old, new, oldtext, newtext, reverse=False):
    aa=endpoint_ranges(pair,'old',old,oldtext,reverse);bb=endpoint_ranges(pair,'new',new,newtext,reverse)
    compared=[]
    for b in bb:
        matches=[a for a in aa if a['label']==b['label'] and a['status']=='resolved_static_numbered_list']
        if len(matches)!=1 or b['status']!='resolved_static_numbered_list':continue
        a=matches[0];mapping=[]
        oldkeys={i:item_key(v['quote']) for i,v in a['items'].items()}
        for number in b['selected_numbers']:
            item=b['items'][number];key=item_key(item['quote'])
            exact=[i for i,k in oldkeys.items() if k==key]
            row={'new_number':number,'new_item':item,'status':'unmatched_selected_item'}
            if len(exact)==1:
                prior=exact[0];selected=prior in a['selected_numbers']
                row.update(status='same_literal_selected_item' if selected else 'newly_referenced_existing_literal_item',
                           old_number=prior,old_item=a['items'][prior],old_was_selected=selected)
            else:
                ranked=[]
                for i,k in oldkeys.items():
                    score,basis=text_match_score(item['quote'],a['items'][i]['quote'],key,k)
                    ranked.append((score,i,basis))
                ranked.sort(reverse=True)
                row['possible_text_matches']=[dict(old_number=i,old_was_selected=i in a['selected_numbers'],text_similarity=round(score,4),
                     similarity_basis=basis,old_item=a['items'][i],certified_correspondence=False) for score,i,basis in ranked[:2] if score>=.5]
            mapping.append(row)
        compared.append({'label':b['label'],'old_numbers':a['selected_numbers'],'new_numbers':b['selected_numbers'],
            'mapping':mapping,'mathematical_direction':'unknown','new_condition_count':None})
    return {'old':aa,'new':bb,'comparisons':compared,
            'scope':'explicit_bounded_numbered_lists_text_matching_only_not_global_assumption_or_entailment_check'}
