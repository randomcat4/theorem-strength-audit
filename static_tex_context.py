"""Conservative source-backed literal aliases, without executing TeX."""
import re
from hashlib import sha256

VERSION='1.0.0'
MAX_MACROS=128
_DEFINITION=re.compile(r'\\(?P<kind>newcommand|renewcommand|providecommand|def)\*?\s*(?:\{(?P<braced>\\[A-Za-z]+)\}|(?P<plain>\\[A-Za-z]+))\s*(?:(?:\[(?P<arity>\d)\])|(?P<defargs>(?:\#\d)*))\s*')

def group(text,start):
    if start>=len(text) or text[start]!='{':return None
    depth=0
    for i in range(start,min(len(text),start+12000)):
        if i and text[i-1]=='\\':continue
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[start+1:i],i+1
    return None

def _outer(body):
    body=body.strip()
    for _ in range(6):
        g=group(body,0)
        if not g or g[1]!=len(body):break
        body=g[0].strip()
    return body

def supported(body,arity):
    body=_outer(body)
    if arity==1:
        value=re.sub(r'\\color\s*\{[A-Za-z]+\}','',body)
        value=re.sub(r'\\textcolor\s*\{[A-Za-z]+\}\s*\{\s*#1\s*\}', '#1',value)
        return 'color_identity_wrapper' if _outer(value).replace(' ','')=='#1' else None
    if arity==0 and (re.fullmatch(r'\\mathbb\s*(?:\{[NZQRCF]\}|[NZQRCF])',body)
         or re.fullmatch(r'\\(?:alpha|beta|gamma|delta|epsilon|varepsilon|theta|vartheta|lambda|mu|nu|xi|pi|rho|sigma|tau|phi|varphi|psi|omega)',body)):
        return 'literal_symbol_alias'
    return None

def extract_context(source,statement_start):
    prefix=source[:statement_start] if isinstance(statement_start,int) else ''
    prefix=re.sub(r'(?<!\\)%[^\n\r]*',lambda m:' '*len(m.group()),prefix)
    definitions={};counts={}
    # Conditional definitions are not evaluated. Reject definitions whose last
    # conditional opener has not been closed, including custom \\if... tokens.
    condition_tokens=list(re.finditer(r'\\(?:if[A-Za-z]*|fi)(?![A-Za-z])',prefix))
    conditional_depth=0;cursor=0
    for m in _DEFINITION.finditer(prefix):
        while cursor<len(condition_tokens) and condition_tokens[cursor].start()<m.start():
            token=condition_tokens[cursor].group()
            conditional_depth=max(0,conditional_depth-1) if token==r'\fi' else conditional_depth+1
            cursor+=1
        name=m['braced'] or m['plain'];counts[name]=counts.get(name,0)+1
        arity=int(m['arity']) if m['arity'] is not None else len(m['defargs'] or '')//2
        g=group(prefix,m.end())
        if not g or conditional_depth:continue
        body=g[0];kind=supported(body,arity)
        if kind:
            definitions[name]={'arity':arity,'body':body,'rule':kind,'line':source.count('\n',0,m.start())+1,
                  'definition_quote':source[m.start():g[1]],'definition_sha256':sha256(source[m.start():g[1]].encode()).hexdigest()}
    accepted={k:v for k,v in sorted(definitions.items()) if counts[k]==1}
    return {'version':VERSION,'macros':dict(list(accepted.items())[:MAX_MACROS]),
         'coverage':'literal supported definitions before located statement only; duplicates and conditional definitions rejected',
         'statement_located':isinstance(statement_start,int),'tex_executed':False,
         'unsupported_or_redefined_names':sum(k not in accepted for k in counts)}

def project(text,context):
    macros=context.get('macros',{}) if isinstance(context,dict) else {}
    used=set()
    for _ in range(5):
        changed=False;parts=[];last=0
        for m in re.finditer(r'\\[A-Za-z]+',text):
            if m.start()<last:continue
            rule=macros.get(m.group(),{})
            if not isinstance(rule,dict) or not supported(rule.get('body',''),rule.get('arity')):continue
            end=m.end()
            if rule['arity']==0:
                replacement=_outer(rule['body'])
                # Preserve TeX word boundaries after a substituted control word.
                if re.fullmatch(r'\\[A-Za-z]+',replacement):replacement+=' '
            else:
                while end<len(text) and text[end].isspace():end+=1
                g=group(text,end)
                if not g:continue
                replacement=g[0];end=g[1]
            parts.extend([text[last:m.start()],replacement]);last=end
            used.add(m.group());changed=True
        if not changed:break
        parts.append(text[last:]);text=''.join(parts)
        if len(text)>96000:break
    return text,sorted(used)
