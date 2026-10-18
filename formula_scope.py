"""Bounded syntactic analysis of formula conditions and domains.

Only finite interval/standard-set rules determine a LOCAL containment direction.
This is neither a TeX executor nor a theorem prover. No files, model decisions,
paper identifiers or evaluation labels are used by this module.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import json
import re
from static_tex_context import project, group

VERSION = '0.4.4'
MAX_ATOMS = 160
MAX_FINDINGS = 40
_REL = re.compile(r'\\(?:notin|subseteq|subsetneq|subset|neq|leqslant|geqslant|leq|geq|forall|exists|in|ne|le|ge)(?![A-Za-z])|<=|>=|!=|[≤≥≠∈<>=]')
_CANON = {'\\leqslant':'<=', '\\leq':'<=', '\\le':'<=', '≤':'<=',
          '\\geqslant':'>=', '\\geq':'>=', '\\ge':'>=', '≥':'>=',
          '\\neq':'!=', '\\ne':'!=', '≠':'!=', '\\in':'in', '∈':'in', '\\notin':'notin'}
_SIMPLE = r'(?:\\[A-Za-z]+|[A-Za-z])(?:_(?:\{[^{}]{1,60}\}|[A-Za-z0-9])|\^(?:\{[^{}]{1,60}\}|[A-Za-z0-9])|\x27)*'
_NUM = r'[+-]?(?:\d+(?:\.\d+)?|\\frac\{[+-]?\d+\}\{\d+\})'
_ASSUMPTION_ROLES = {'premise', 'declaration', 'universal_domain', 'postposed_domain'}
_LABELS = {
 'tightened':'同一对象的局部条件收紧', 'relaxed':'同一对象的局部条件放宽',
 'replaced':'同一对象的条件替换，集合包含方向未算出',
 'added_condition':'同一词面对象新增条件公式',
 'newly_explicit_domain':'同一参数补写了所属集合，需核对原有约定',
 'removed_condition':'原条件公式未再出现',
 'formula_role_changed':'相同公式移到不同句法位置',
 'reference_scope_changed':'引用的条件清单变化，需核对内容及编号',
 'quantifier_scope_changed':'量词或常数依赖的句法范围变化',
 'condition_expression_changed':'条件公式的左端表达式变化，暂不比较集合包含方向',
}

def normalize(value):
    value = re.sub(r'(?<!\\)%[^\r\n]*', '', value)
    value = re.sub(r'\\(?:left|right|[bB]igg?[lrm]?|displaystyle|textstyle|limits|nolimits|quad|qquad)(?![A-Za-z])', '', value)
    value = re.sub(r'\\[,;!: ]', '', value)
    value = re.sub(r'(?<!\\)&', '', value)
    value = re.sub(r'\\(?:begin|end)\{(?:split|aligned|gathered)\}', '', value)
    value = re.sub(r'\\color\s*\{[A-Za-z]+\}', '', value)
    value = re.sub(r'\\(?:label|tag|eqref|ref|cite\w*)\s*\{[^{}]*\}', '', value)
    value = value.replace(r'\dfrac',r'\frac').replace(r'\tfrac',r'\frac')
    # Braces for a single sub/superscript token do not change its operand.
    value = re.sub(r'([_^])\{([A-Za-z0-9]|\\[A-Za-z]+)\}', r'\1\2', value)
    # TeX control-word termination matters: "\\in X" must not become "\\inX".
    value = re.sub(r'(\\[A-Za-z]+)\s+(?=[A-Za-z])',r'\1@@SPACE@@',value)
    return re.sub(r'\s+', '', value).replace('@@SPACE@@',' ').strip('.,;')

def _number(value):
    value = value.strip()
    m = re.fullmatch(r'([+-]?)\\frac\{([+-]?\d+)\}\{(\d+)\}', value)
    try:
        if m:
            return Fraction(int((m[1] or '')+m[2]), int(m[3]))
        return Fraction(value) if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', value) else None
    except (ValueError, ZeroDivisionError):
        return None

def _strip_outer(value):
    for _ in range(8):
        if len(value) < 2 or value[0]+value[-1] not in {'{}','()'}:
            break
        depth = 0
        complete = True
        for i,c in enumerate(value):
            depth += c == value[0]
            depth -= c == value[-1]
            if depth == 0 and i != len(value)-1:
                complete = False
                break
        if not complete:
            break
        value = value[1:-1]
    return value

def _standard_set(value):
    m = re.fullmatch(r'\\mathbb\s*(?:\{([NZQRC])\}|([NZQRC]))', value)
    return (m[1] or m[2]) if m else None

def _domain_direction(a,b):
    a,b=_strip_outer(a),_strip_outer(b)
    if a == b:
        return 'equivalent', '同一规范化集合文字'
    # This order only asserts containment, not conventions on whether 0 is in N.
    sa,sb = _standard_set(a),_standard_set(b)
    if sa and sb:
        if sa==sb:return 'equivalent','同一标准数集的两种 TeX 字面写法'
        order = 'NZQRC'
        return ('tightened' if order.index(sb)<order.index(sa) else 'relaxed'), '标准数集包含链 N⊆Z⊆Q⊆R⊆C；不解释 N 是否含零'
    # Products and equal written exponents preserve componentwise inclusion.
    pa,pb=a.split(r'\times'),b.split(r'\times')
    if len(pa)==len(pb) and len(pa)>1:
        directions=[_domain_direction(x,y)[0] for x,y in zip(pa,pb)]
        if set(directions)=={'equivalent'}:return 'equivalent','同一标准数集积的字面换写'
        if set(directions)<={'tightened','equivalent'}:return 'tightened','标准数集积的逐分量包含，保留相同指数'
        if set(directions)<={'relaxed','equivalent'}:return 'relaxed','标准数集积的逐分量包含，保留相同指数'
    aa=re.fullmatch(r'(.+)(\^(?:\{[^{}]+\}|[^{}]+))',a)
    bb=re.fullmatch(r'(.+)(\^(?:\{[^{}]+\}|[^{}]+))',b)
    if aa and bb and aa[2]==bb[2] and _standard_set(_strip_outer(aa[1])) and _standard_set(_strip_outer(bb[1])):
        direction,basis=_domain_direction(aa[1],bb[1]);return direction,basis+'；同一词面指数的乘积'
    # Explicit deletion of elements from the SAME written base set.
    for base,restricted,direction in ((a,b,'tightened'),(b,a,'relaxed')):
        if restricted.startswith(base+r'\setminus') or restricted.startswith(base+r'\backslash'):
            tail = restricted[len(base):]
            if re.fullmatch(r'\\(?:setminus|backslash)\\?\{[^{}]{1,90}\\?\}',tail):
                return direction,'同一词面集合显式作差；只断言包含，不断言真子集'
    ia,ib = _interval(a),_interval(b)
    if ia and ib:
        return _interval_direction(ia,ib), '区间端点与开闭性比较；不执行符号代数'
    return 'replaced','未使用可认证的有限集合包含规则'

def _interval(value):
    m = re.fullmatch(r'([\[(])([^,]{1,80}),([^,]{1,80})([\])])', value)
    if not m:
        return None
    return {'lower':m[2], 'lower_closed':m[1]=='[', 'upper':m[3], 'upper_closed':m[4]==']'}

def _cmp(a,b):
    if a == b:
        return 0
    infinities = {'-\\infty':-1,'-∞':-1,'\\infty':1,'+\\infty':1,'∞':1}
    if a in infinities or b in infinities:
        return -infinities[b] if a not in infinities else infinities[a] if b not in infinities else (infinities[a]>infinities[b])-(infinities[a]<infinities[b])
    na,nb = _number(a),_number(b)
    return (na>nb)-(na<nb) if na is not None and nb is not None else None

def _subset_interval(a,b):
    lo,hi = _cmp(a['lower'],b['lower']),_cmp(a['upper'],b['upper'])
    if lo is None or hi is None:
        return None
    return (lo>0 or lo==0 and (not a['lower_closed'] or b['lower_closed'])) and (hi<0 or hi==0 and (not a['upper_closed'] or b['upper_closed']))

def _interval_direction(a,b):
    # Empty/reversed ranges need a separate treatment; avoid inferring from them.
    for value in (a,b):
        order = _cmp(value['lower'],value['upper'])
        if order is not None and (order>0 or order==0 and not(value['lower_closed'] and value['upper_closed'])):
            return 'replaced'
    new_in_old,old_in_new = _subset_interval(b,a),_subset_interval(a,b)
    if new_in_old is True and old_in_new is True:
        return 'equivalent'
    if new_in_old is True:
        return 'tightened'
    if old_in_new is True:
        return 'relaxed'
    return 'replaced'

def _mask_math(text,units):
    chars=list(text)
    for u in units:
        for p in range(u['start'],min(len(chars),u['end'])):
            chars[p]=' '
    return ''.join(chars)

def _role(text,unit,prose):
    start = unit['start']
    left = max([0]+[m.end() for m in re.finditer(r'[.!?](?=\s|$)|\\item(?:\[[^\]]*\])?',prose[:start])])
    prefix = prose[left:start][-900:]
    # Iff conditions are operands of an equivalence, not automatically premises.
    if re.search(r'\bif\s+and\s+only\s+if\b|\biff\b',prefix,re.I):
        return 'biconditional_operand'
    markers=list(re.finditer(r'\b(?:let|suppose|assuming|assume|provided|whenever|when|if|then)\b|\bwe\s+have\b|\bthere\s+exists?\b|\bthere\s+is\s+(?:a|an)\s+constant\b|\bfor\s+(?:(?:any|all|every|each)\b)?',prefix,re.I))
    if markers:
        last=markers[-1].group().lower().strip()
        if last.startswith('for'):
            suffix=prefix[markers[-1].end():]
            if re.search(r'\b(?:have|holds?|is|are|bounded|given|following|satisfies|equations?|systems?|obtain|estimates?|we)\b',suffix,re.I):
                return 'conclusion'
            return 'universal_domain' if re.search(r'any|all|every|each',last) else 'postposed_domain'
        if last in {'then','we have'} or last.startswith('there exist') or last.startswith('there is'):
            return 'conclusion'
        return 'declaration' if last=='let' and not re.search(r'\b(?:satisfy|satisfying|such\s+that|with)\b',prefix[markers[-1].end():],re.I) else 'premise'
    if re.search(r'\\forall(?![A-Za-z])|∀',unit['normalized']):
        return 'universal_domain'
    if re.search(r'\\exists(?![A-Za-z])|∃',unit['normalized']):
        return 'existential_operand'
    return 'unscoped_formula'

def _split_top(value, separators=',;'):
    parts=[]
    depth=0
    start=0
    for i,c in enumerate(value):
        if c in '{([':
            depth+=1
        elif c in '})]':
            depth=max(0,depth-1)
        elif c in separators and depth==0:
            parts.append(value[start:i]);start=i+1
    parts.append(value[start:])
    return parts

def _relation_atoms(expr):
    """Relational chains, element membership and standard intervals; no eval()."""
    expr=normalize(expr)
    expr=re.sub(r'\\(?:text|textrm|mbox)\{(?:if|for|when|whenever|and)\}',',',expr,flags=re.I)
    expr=re.sub(r'\\(?:quad|qquad|land|wedge)(?![A-Za-z])',',',expr)
    expr=re.sub(r'\\forall(?![A-Za-z])|∀','',expr)
    if re.search(r'\\(?:exists|lor|vee|iff|Leftrightarrow)(?![A-Za-z])|∃|\botherwise\b',expr):
        return [],['existential_disjunctive_or_biconditional_formula']
    finite_membership=bool(re.fullmatch(_SIMPLE+r'\\in\s*[^=<>]{1,220}',expr) and r'\mid' not in expr and '|' not in expr)
    if re.search(r'\\(?:sum|prod|int|sup|min|max)(?![A-Za-z])|\\begin\{(?:cases|array)',expr) or r'\{' in expr and not finite_membership:
        return [],['nested_set_operator_or_cases_not_flat_condition']
    if r'\coloneqq' in expr or ':=' in expr or r'\colon' in expr:
        return [],['definition_or_map_not_assumption']
    if re.search(r'\\(?:dots|ldots|cdots)(?![A-Za-z])',expr):
        return [],['ellipsis_range_not_a_single_point_constraint']
    findings=[];unsupported=[]
    # a,b in S: keep the repeated domain but separate textual objects.
    expr=re.sub(r'(?P<objs>'+_SIMPLE+r'(?:,'+_SIMPLE+r')+)\\in\s*(?P<set>[^,;]+)',
                lambda m:';'.join(o+r'\in '+m['set'] for o in m['objs'].split(',')),expr)
    expr=re.sub(r'(?P<objs>'+_SIMPLE+r'(?:,'+_SIMPLE+r')+)(?P<rel>\\(?:leq|geq|le|ge|neq|ne)(?![A-Za-z])|<=|>=|!=|[<>=])(?P<value>'+_NUM+r')(?=;|$)',
                lambda m:';'.join(o+m['rel']+' '+m['value'] for o in m['objs'].split(',')),expr)
    for part in _split_top(expr):
        part=_strip_outer(part.strip())
        if not part:
            continue
        tokens=list(_REL.finditer(part))
        # Only relations at ordinary top level. A subscript bound is not a claim.
        flat=[]
        for m in tokens:
            depth=0
            for i,c in enumerate(part[:m.start()]):
                if c in '{([' and (i==0 or part[i-1]!='\\'):depth+=1
                elif c in '})]' and (i==0 or part[i-1]!='\\'):depth-=1
            if depth==0 and m.group() not in {r'\forall',r'\exists'}:
                flat.append(m)
        if not flat:
            if _REL.search(part):unsupported.append('relation_nested_or_not_flat')
            continue
        operands=[];last=0
        for m in flat:
            operands.append(part[last:m.start()]);last=m.end()
        operands.append(part[last:])
        for i,m in enumerate(flat):
            lhs,rhs=operands[i].strip(),operands[i+1].strip()
            rel=_CANON.get(m.group(),m.group())
            if not lhs or not rhs:continue
            if re.fullmatch(_SIMPLE,lhs) and rel in {'in','notin',r'\subset',r'\subseteq',r'\subsetneq'}:
                findings.append({'object':lhs,'family':'domain','relation':rel,'value':rhs,'canonical':lhs+rel+rhs})
                continue
            if rel in {'<','<=','>','>=','=','!='}:
                # Reverse a numeric left operand, e.g. 0 < p < 1.
                if _number(lhs) is not None and _number(rhs) is None:
                    lhs,rhs=rhs,lhs
                    rel={'<':'>','<=':'>=','>':'<','>=':'<=','=':'=','!=':'!='}[rel]
                if re.fullmatch(_SIMPLE,lhs) or re.fullmatch(r'[A-Za-z\\][^=<>]{0,150}',lhs):
                    findings.append({'object':lhs,'family':'bound' if _number(rhs) is not None else 'relation',
                                     'relation':rel,'value':rhs,'canonical':lhs+rel+rhs})
    return findings,unsupported

def _inventory(text,units,evidence,macros=None):
    prose=_mask_math(text,units)
    values=[];unsupported=Counter()
    for index,u in enumerate(units[:80]):
        role=_role(text,u,prose)
        expression=project(u.get('raw_core',u['normalized']),macros)[0]
        parts=[(expression,role)]
        if u.get('condition'):
            parts.append((project(u['condition'],macros)[0],'branch_domain'))
        for expr,local_role in parts:
            # An explicit implication has a syntactic antecedent/consequent.
            implication=re.split(r'\\(?:implies|Rightarrow|Longrightarrow)(?![A-Za-z])|⇒',expr)
            positioned=[(implication[0],'premise'),(implication[1],'conclusion')] if len(implication)==2 else [(expr,local_role)]
            for fragment,fragment_role in positioned:
                atoms,notes=_relation_atoms(fragment)
                unsupported.update(notes)
                ev=evidence(text,u)
                nonconjunctive=bool(re.search(r'\b(?:or|unless|not|neither|either)\b',prose[slice(*ev['context_span'])],re.I))
                for atom in atoms:
                    actual_role=fragment_role
                    if actual_role in {'universal_domain','postposed_domain'} and not re.fullmatch(_SIMPLE,atom['object']):
                        actual_role='conclusion'
                    if actual_role=='declaration' and atom['relation']=='=':
                        actual_role='definition_or_fixed_parameter'
                    atom.update(role=fragment_role if not nonconjunctive else 'nonconjunctive_scope',
                         formula_index=index, branch_index=u.get('row_index') if fragment_role=='branch_domain' else None,
                         clause_id=len(re.findall(r'\\item(?![A-Za-z])',text[:u['start']])),
                         span=ev['formula_span'], quote=ev['formula_quote'], context_quote=ev['context_quote'],
                         context_span=ev['context_span'])
                    atom['role']=actual_role if not nonconjunctive else 'nonconjunctive_scope'
                    atom['guarantee_scope']=_guarantee_scope(text,u,index,units,prose,atom['role'])
                    values.append(atom)
                    if len(values)>=MAX_ATOMS:
                        unsupported['atom_limit_reached']+=1
                        return values,dict(unsupported)
    return values,dict(unsupported)

def _guarantee_scope(text,unit,index,units,prose,role):
    """Locate a local formula target; never promote it to whole-theorem scope."""
    clause=lambda u:len(re.findall(r'\\item(?![A-Za-z])',text[:u['start']]))
    own_clause=clause(unit)
    displayed=lambda u:u.get('environment') not in {'dollar_inline','paren_inline'}
    earlier=[u for u in units[:index] if clause(u)==own_clause and displayed(u)]
    later=[u for u in units[index+1:] if clause(u)==own_clause and displayed(u)]
    def target(u):
        return {'formula_quote':text[u['start']:u['end']], 'formula_span':[u['start'],u['end']],
                'tex_labels':re.findall(r'\\label\s*\{([^{}]+)\}',text[u['start']:u['end']]),
                'source_basis':'nearest_displayed_formula_in_same_enumerated_item_not_full_logical_scope'}
    if role=='branch_domain':
        return {'extent':'one_formula_branch','label':'仅此分段公式的局部条件','target':target(unit),
                'whole_statement_scope_certified':False}
    item_start=max([0]+[m.end() for m in re.finditer(r'\\item(?:\[[^\]]*\])?',prose[:unit['start']])])
    prefix=prose[item_start:unit['start']]
    local_marker=list(re.finditer(r'\bsuch\s+that\s+for\s+(?:any|all|every|each)\b|\bwe\s+have\s+for\s+(?:any|all|every|each)\b|\b(?:when|whenever|if)\b',prefix,re.I))
    if role in _ASSUMPTION_ROLES and later:
        local=bool(earlier and local_marker and local_marker[-1].start()>earlier[-1]['end']-item_start)
        return {'extent':'local_following_formula' if local else 'same_item_following_formula',
                'label':'后续这一条估计/保证的条件，不能外推为整条命题的条件' if local else '同分项后续公式的条件线索；完整绑定范围未认证',
                'target':target(later[0]),'earlier_formula_in_same_item':target(earlier[-1]) if earlier else None,
                'whole_statement_scope_certified':False}
    return {'extent':'not_resolved','label':'尚未自动定位到唯一保证公式；不声称作用于整条命题',
            'target':None,'whole_statement_scope_certified':False}

def _bounds_interval(atoms):
    lower=('-\\infty',False);upper=('\\infty',False)
    for a in atoms:
        rel,value=a['relation'],a['value']
        if rel=='!=':return None
        if rel in {'>','>=','='}:
            cmp=_cmp(value,lower[0])
            if cmp is None:return None
            if cmp>0:lower=(value,rel in {'>=','='})
            elif cmp==0:lower=(value,lower[1] and rel in {'>=','='})
        if rel in {'<','<=','='}:
            cmp=_cmp(value,upper[0])
            if cmp is None:return None
            if cmp<0:upper=(value,rel in {'<=','='})
            elif cmp==0:upper=(value,upper[1] and rel in {'<=','='})
    return {'lower':lower[0],'lower_closed':lower[1],'upper':upper[0],'upper_closed':upper[1]}

def _object_present(text,obj):
    # Exact symbol tokens; x is not found in \exp, X, x_1, or an English word.
    return bool(re.search(r'(?<![A-Za-z0-9_\\])'+re.escape(obj)+r'(?![A-Za-z0-9_])',normalize(text)))

def _finding(kind,old_atoms,new_atoms,basis,eligible=False):
    first=(new_atoms or old_atoms)[0]
    old_quote=' ; '.join(dict.fromkeys(a['quote'] for a in old_atoms))
    new_quote=' ; '.join(dict.fromkeys(a['quote'] for a in new_atoms))
    label=_LABELS[kind]
    if kind in {'added_condition','removed_condition'} and first['role'] not in _ASSUMPTION_ROLES:
        label='该位置新增关系式' if kind=='added_condition' else '该位置原关系式未再出现'
    return {'kind':kind,'label':label,'object_key':first['object'],'syntactic_role':first['role'],
       'clause_id':first['clause_id'],
       'old_guarantee_scopes':[a['guarantee_scope'] for a in old_atoms],
       'new_guarantee_scopes':[a['guarantee_scope'] for a in new_atoms],
       'old_atoms':[a['canonical'] for a in old_atoms],'new_atoms':[a['canonical'] for a in new_atoms],
       'old_quote':old_quote,'new_quote':new_quote,'old_span':old_atoms[0]['span'] if old_atoms else None,
       'new_span':new_atoms[0]['span'] if new_atoms else None,
       'old_spans':[list(s) for s in dict.fromkeys(tuple(a['span']) for a in old_atoms)],
       'new_spans':[list(s) for s in dict.fromkeys(tuple(a['span']) for a in new_atoms)],
       'quote_span_semantics':'quotes may join several formula spans with semicolons; singular span is the first formula only',
       'old_context':old_atoms[0]['context_quote'] if old_atoms else None,
       'new_context':new_atoms[0]['context_quote'] if new_atoms else None,
       'comparison_rule':basis, 'local_condition_direction':kind if kind in {'tightened','relaxed'} else 'not_established',
       'eligible_as_added_premise':eligible,'global_weakening_certified':False,
       'global_assumptions_checked':False,'object_identity':'case_sensitive_textual_object_only',
       'span_unit':'characters_within_extracted_statement',
       'finding_id':'scope_'+sha256((kind+'\0'+first['object']+'\0'+old_quote+'\0'+new_quote).encode()).hexdigest()[:20]}

def _reference_changes(old,new,old_units,new_units):
    # Retain raw references. A longer number list is not proof of extra content.
    pattern=re.compile(r'\b(?:regularity|assumptions?|conditions?|hypotheses)\b[^.;]{0,220}',re.I)
    def signatures(text,units):
        values=[]
        prose=_mask_math(text,units)
        for m in pattern.finditer(prose):
            raw=text[m.start():m.end()]
            refs=re.findall(r'\\(?:ref|eqref|cref)\{([^{}]+)\}',m.group())
            numbered_prose=re.sub(r'\\(?:ref|eqref|cref|cite\w*)\{[^{}]*\}','',m.group())
            nums=re.findall(r'\((\d{1,3})\)',numbered_prose)
            if nums or refs:values.append((tuple(nums),tuple(refs),raw,[m.start(),m.end()]))
        return values
    a,b=signatures(old,old_units),signatures(new,new_units)
    result=[]
    if [(v[0],v[1]) for v in a] != [(v[0],v[1]) for v in b] and (a or b):
        result.append({'kind':'reference_scope_changed','label':_LABELS['reference_scope_changed'],
            'old_quote':' ; '.join(v[2] for v in a),'new_quote':' ; '.join(v[2] for v in b),
            'old_span':a[0][3] if a else None,'new_span':b[0][3] if b else None,
            'old_reference_numbers':[list(v[0]) for v in a], 'new_reference_numbers':[list(v[0]) for v in b],
            'local_condition_direction':'not_established','eligible_as_added_premise':False,
            'comparison_rule':'可见条件编号/标签清单不同；未把重新编号当作新增前提',
            'referenced_content_resolved':False,'global_weakening_certified':False})
    return result

def analyze_formula_scope(old,new,math_units,formula_evidence,old_macros=None,new_macros=None):
    au,bu=math_units(old),math_units(new)
    aa,an=_inventory(old,au,formula_evidence,old_macros);ba,bn=_inventory(new,bu,formula_evidence,new_macros)
    groups_a=defaultdict(list);groups_b=defaultdict(list)
    for atom,target in [(a,groups_a) for a in aa]+[(a,groups_b) for a in ba]:
        target[(atom['object'],atom['role'],atom['family'],(atom['clause_id'],atom['branch_index']))].append(atom)
    findings=[]
    for key in sorted(groups_a.keys()|groups_b.keys(),key=repr):
        a,b=groups_a[key],groups_b[key]
        ca,cb=Counter(v['canonical'] for v in a),Counter(v['canonical'] for v in b)
        if ca==cb:continue
        role=key[1];basis='同一词面对象、同一句法角色；未作全局蕴含推导'
        eligible=role in _ASSUMPTION_ROLES
        if a and b:
            if key[2]=='bound' and len({v['formula_index'] for v in a})<=2 and len({v['formula_index'] for v in b})<=2:
                ia,ib=_bounds_interval(a),_bounds_interval(b)
                direction=_interval_direction(ia,ib) if ia and ib else 'replaced'
                basis='同一词面对象的数值上下界取交并比较端点；'+basis
            elif key[2]=='domain' and len(a)==len(b)==1 and a[0]['relation']==b[0]['relation'] and a[0]['relation']!='notin':
                direction,basis=_domain_direction(a[0]['value'],b[0]['value'])
            elif len(a)==len(b)==1 and a[0]['relation']=='!=' and b[0]['relation']=='!=':
                direction='replaced'
            else:
                direction='replaced'
            if direction=='equivalent':continue
            findings.append(_finding(direction,a,b,basis,eligible and direction=='tightened'))
        elif b and _object_present(old,key[0]):
            kind='newly_explicit_domain' if key[2]=='domain' else 'added_condition'
            # A condition moved from a different syntactic role is not added.
            moved=any(v['canonical']==w['canonical'] and v['clause_id']==w['clause_id'] for v in aa for w in b)
            if moved:
                kind='formula_role_changed';eligible=False
            findings.append(_finding(kind,[],b,basis,eligible and kind=='added_condition' and bool(re.fullmatch(_SIMPLE,key[0]))))
        elif a and _object_present(new,key[0]):
            moved=any(v['canonical']==w['canonical'] and v['clause_id']==w['clause_id'] for v in ba for w in a)
            findings.append(_finding('formula_role_changed' if moved else 'removed_condition',a,[],basis,False))
        if len(findings)>=MAX_FINDINGS:break
    # Detect a unique changed condition expression even when its left side no
    # longer matches; no added-premise promotion or algebraic comparison follows.
    old_keys={(a['canonical'],a['clause_id']) for a in aa}
    new_keys={(a['canonical'],a['clause_id']) for a in ba}
    residual_a=[a for a in aa if (a['canonical'],a['clause_id']) not in new_keys and a['role'] in _ASSUMPTION_ROLES]
    residual_b=[a for a in ba if (a['canonical'],a['clause_id']) not in old_keys and a['role'] in _ASSUMPTION_ROLES]
    for a in residual_a:
        compatible=lambda x,y:x['relation']==y['relation'] and x['value']==y['value'] and x['clause_id']==y['clause_id'] and x['object']!=y['object']
        candidates=[b for b in residual_b if compatible(a,b)]
        if len(candidates)==1 and sum(compatible(v,candidates[0]) for v in residual_a)==1:
            b=candidates[0]
            findings.append(_finding('condition_expression_changed',[a],[b],
                 '同一分项中关系号和右端相同，左端表达式不同；保留两段原式，不认证对象相同或强弱方向',False))
    findings.extend(_reference_changes(old,new,au,bu))
    qpattern=re.compile(r'\b(?:for\s+(?:any|every|all|each)|there\s+exists?|there\s+is\s+(?:a|an)\s+constant|uniform(?:ly)?)\b|\\(?:forall|exists)(?![A-Za-z])',re.I)
    quant=lambda t:[m.group() for m in qpattern.finditer(t)]
    aq,bq=quant(old),quant(new)
    if [q.lower() for q in aq]!=[q.lower() for q in bq]:
        findings.append({'kind':'quantifier_scope_changed','label':_LABELS['quantifier_scope_changed'],
             'old_quote':' → '.join(old[m.start():min(len(old),m.end()+100)].split('.')[0] for m in qpattern.finditer(old)),
             'new_quote':' → '.join(new[m.start():min(len(new),m.end()+100)].split('.')[0] for m in qpattern.finditer(new)),
             'old_span':None,'new_span':None,
             'eligible_as_added_premise':False,'local_condition_direction':'not_established',
             'comparison_rule':'量词/一致性标记出现次序变化；没有认证变量绑定及常数依赖',
             'global_weakening_certified':False})
    return {'version':VERSION,'findings':findings[:MAX_FINDINGS],
         'formula_edit_ledger':_formula_edit_ledger(old,new,au,bu,old_macros,new_macros),
         'atom_counts':{'old':len(aa),'new':len(ba)}, 'parsed_roles':dict(Counter(a['role'] for a in aa+ba)),
         'unsupported_syntax':{'old':an,'new':bn},
         'local_direction_count':sum(v.get('local_condition_direction') in {'tightened','relaxed'} for v in findings),
         'runtime_boundary':'finite deterministic syntax and interval/set rules; no files, model or annotations',
         'global_mathematical_direction':'not_certified'}

def _formula_edit_ledger(old,new,au,bu,old_context,new_context):
    """Inventory actual exponent/index/macro tokens; no assumed formula pairing."""
    def inventory(text,units,context):
        values={key:Counter() for key in ('exponents','subscripts','macros','numbers')}
        for u in units:
            raw=normalize(project(u.get('raw_core',u['normalized']),context)[0])
            values['macros'].update(re.findall(r'\\[A-Za-z]+',raw))
            values['numbers'].update(re.findall(r'\d+(?:\.\d+)?',raw))
            for m in re.finditer(r'[_^]',raw):
                rest=raw[m.end():]
                g=group(raw,m.end())
                atom=re.match(r'\\[A-Za-z]+|.',rest)
                operand=g[0] if g else atom.group() if atom else ''
                values['exponents' if m.group()=='^' else 'subscripts'][operand]+=1
        return values
    a,b=inventory(old,au,old_context),inventory(new,bu,new_context)
    result={}
    for key in a:
        result[key]={'removed':[{'text':k,'count':v} for k,v in sorted((a[key]-b[key]).items())[:24]],
                     'added':[{'text':k,'count':v} for k,v in sorted((b[key]-a[key]).items())[:24]],
                     'multiset_changed':a[key]!=b[key]}
    result['meaning']='仅记录公式中指数、下标、宏及数值词元的变化与重复次数；没有把不同公式硬配或判断代数等价。'
    return result

def condition_rewrite_projection(text,math_units,formula_evidence,context=None):
    text,used=project(text,context)
    type_rewrites=len(re.findall(r'\bsmooth\s+complex\s+manifolds?\b',text,re.I))
    # A complex manifold's differentiable smoothness is inherent in this type;
    # this does not erase "smooth" from schemes, analytic spaces or manifolds in
    # general. Negations remain in the projected whole statement.
    text=re.sub(r'\bsmooth\s+(complex\s+manifolds?)\b',r'\1',text,flags=re.I)
    spelling_rewrites=len(re.findall(r'\btransversally\b',text,re.I))
    text=re.sub(r'\btransversally\b','transversely',text,flags=re.I)
    units=math_units(text)
    atoms,_=_inventory(text,units,formula_evidence)
    grouped=defaultdict(list)
    for atom in atoms:grouped[atom['formula_index']].append(atom)
    replacements=[]
    for i,items in grouped.items():
        # Whole formula must be exactly a flat constraint on one simple object.
        if len({a['object'] for a in items})!=1 or not re.fullmatch(_SIMPLE,items[0]['object']):continue
        raw=normalize(units[i].get('raw_core',units[i]['normalized']))
        if re.search(r'\\(?:forall|exists)(?![A-Za-z])|∀|∃',raw):continue
        interval=None
        if all(a['family']=='bound' for a in items):
            interval=_bounds_interval(items)
        elif len(items)==1 and items[0]['family']=='domain' and items[0]['relation']=='in':
            interval=_interval(items[0]['value'])
        if not interval:continue
        # Ensure all content was consumed; prose or arithmetic fragments must not
        # disappear merely because one relation could be extracted from them.
        if not (re.fullmatch(_SIMPLE+r'(?:<=|>=|!=|[<>=]|\\(?:leq|geq|le|ge|neq|ne)(?![A-Za-z]))'+_NUM,raw)
           or re.fullmatch(_NUM+r'(?:<=|>=|[<>]|\\(?:leq|geq|le|ge)(?![A-Za-z]))'+_SIMPLE+r'(?:<=|>=|[<>]|\\(?:leq|geq|le|ge)(?![A-Za-z]))'+_NUM,raw)
           or re.fullmatch(_SIMPLE+r'\\in\s*[\[(][^,]{1,80},[^,]{1,80}[\])]',raw)):
            continue
        for endpoint in ('lower','upper'):
            n=_number(interval[endpoint])
            if n is not None:interval[endpoint]=str(n)
        key=json.dumps([items[0]['object'],interval],sort_keys=True,separators=(',',':'))
        replacements.append((units[i]['start'],units[i]['end'],' LOCALINTERVAL'+key+' '))
    for a,b,value in sorted(replacements,reverse=True):text=text[:a]+value+text[b:]
    return text,{'static_macros_used':used,'flat_intervals_projected':len(replacements),
                 'entailed_type_properties_projected':type_rewrites,'adverb_spelling_variants':spelling_rewrites}

def added_formula_premises(analysis,old,new,math_units=None):
    premises=[]
    for f in analysis['findings']:
        if not f.get('eligible_as_added_premise'):continue
        # Presence anchor is supplied separately when a condition is newly written.
        span=f.get('old_span')
        anchor_kind='old_condition_formula'
        if span is None:
            anchor_kind='same_symbol_formula_context_not_an_old_condition'
            candidates=[]
            if math_units:
                for u in math_units(old):
                    same_clause=len(re.findall(r'\\item(?![A-Za-z])',old[:u['start']]))==f.get('clause_id',0)
                    if same_clause and _object_present(old[u['start']:u['end']],f['object_key']):candidates.append(u)
            span=[candidates[0]['start'],candidates[0]['end']] if candidates else None
            if span is None:anchor_kind='no_precise_old_formula_anchor'
        premises.append({'premise_id':f['finding_id'],'property':'formula_'+f['kind'],
             'label':f['label'],'object_key':f['object_key'],
             'old_minimal':old[slice(*span)] if span else '',
             'new_minimal':new[slice(*f['new_span'])] if f.get('new_span') else '',
             'old_span':span,'new_span':f['new_span'],'span_unit':'characters_within_extracted_statement',
             'old_anchor_kind':anchor_kind,'all_old_condition_spans':f.get('old_spans',[]),
             'all_new_condition_spans':f.get('new_spans',[]),
             'old_condition_formula_present':bool(f.get('old_spans')),
             'new_guarantee_scopes':f.get('new_guarantee_scopes',[]),
             'clause_id':f.get('clause_id'),
             'scope':f['syntactic_role'],'object_alignment':f['object_identity'],
             'association_phrases':[], 'base_type':'formula_condition',
             'logical_strength':'unknown','global_assumptions_checked':False,
             'local_condition_direction':f['local_condition_direction'],
             'comparison_rule':f['comparison_rule']})
    return premises

def classify_change(scope,signals,premises,rewrite,difference):
    findings=scope['findings']
    kinds={s['kind'] for s in signals}
    ledger=scope['formula_edit_ledger']
    if rewrite.get('suppresses_strength_alert'):
        category,label='rewrite_only','有限规则覆盖的纯改写'
    elif premises:
        category,label='premise_change','已定位新增属性或公式条件；查看同时变化的结论'
    elif any(f['kind']=='reference_scope_changed' for f in findings):
        category,label='condition_reference_change','条件引用范围变化；内容及重新编号需核对'
    elif any(f.get('syntactic_role') in _ASSUMPTION_ROLES|{'branch_domain'} for f in findings):
        category,label='formula_scope_change','公式前提、参数所属集合或分支适用范围变化'
    elif any(f['kind']=='quantifier_scope_changed' for f in findings):
        category,label='quantifier_scope_change','量词出现或依赖次序变化；需要查看绑定对象'
    elif any(f.get('syntactic_role')=='conclusion' for f in findings):
        category,label='conclusion_formula_change','结论中的关系式或对象范围发生变化'
    elif any(f.get('local_condition_direction') in {'tightened','relaxed'} for f in findings):
        category,label='formula_scope_change','公式所写集合范围有局部包含变化；当前句法角色尚未定位'
    elif kinds & {'inequality_change','numeric_formula_change','branch_scope_change'}:
        category,label='formula_value_or_relation_change','公式数值、符号或关系式变化；当前句法未给出方向'
    elif ledger['exponents']['multiset_changed']:
        category,label='formula_exponent_change','公式指数词元变化；需核对衰减率、幂次或表示对象'
    elif ledger['subscripts']['multiset_changed']:
        category,label='formula_subscript_change','公式下标词元变化；需核对索引范围和对象'
    elif ledger['macros']['multiset_changed']:
        category,label='formula_macro_change','公式命令或宏文字变化；未由来源定义完整解释'
    elif kinds & {'formula_added','formula_omitted','formula_change','notation_change'}:
        category,label='formula_object_or_notation_change','公式对象、宏或表示结构变化；没有识别为新增前提'
    elif difference.get('status') in {'identical','formatting_only','notation_only'}:
        category,label='no_active_change','当前规则下未见活跃文字变化或仅有表示观察'
    else:
        category,label='prose_or_structure_change','文字、结构或声明角色变化；需要查看具体差异'
    return {'category':category,'label':label,'findings':findings,
         'local_direction_count':scope['local_direction_count'],
         'condition_content_certified':False,'overall_weakening_certified':False,
         'remaining_question':'局部范围可以比较时只说明该范围；混合结论、全局约定、宏及数学等价另行核验。'}
