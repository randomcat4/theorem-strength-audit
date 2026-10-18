"""Added-premise detection on the same textual object: declared properties, modifiers and hypothesis clauses."""

from __future__ import annotations

from hashlib import sha256
import re
from statement_text import _active_text, _math_normalize, _math_units, _normalized, _scope
from alignment import _anchor_words, _conclusion
from premise_clauses import Clause, clauses, body_key, comparison, masks, subject


_QUALIFIERS = (
    ("finite_dimension", "有限维", r"\bfinite[-\s]+dimensional\b|有限维"),
    ("finite_generation", "有限生成", r"\bfinitely[-\s]+generated\b|有限生成"),
    ("finite_presentation", "有限表示", r"\bfinitely[-\s]+presented\b|有限表示"),
    ("finite", "有限", r"\bfinite\b(?![-\s]+dimensional)|有限(?!维|生成|表示)"),
    ("normal", "normal", r"\bnormal\b(?!\s+form)|正规"),
    ("amenable", "可平均", r"\bamenable\b|可平均"),
    ("nonzero", "非零", r"\bnon[-\s]?zero\b|非零"),
    ("nonzero_alternative", "至少一个非零", r"\bat\s+least\s+one\b.{0,90}?\bnon[-\s]?zero\b|至少.{0,12}?非零"),
    ("positive", "正值", r"\bpositive\b|正数|正值"),
    ("compact", "紧致", r"\bcompact\b|紧致"),
    ("connected", "连通", r"\bconnected\b|连通"),
    ("smooth", "光滑", r"\bsmooth\b|光滑"),
    ("separable", "可分", r"\bseparable\b|可分"),
    ("algebraic", "代数", r"\balgebraic\b|代数扩张"),
    ("bounded", "有界", r"\bbounded\b|有界"),
    ("regular", "正则", r"\bregular\b|正则"),
    ("irreducible", "不可约", r"\birreducible\b|不可约"),
    ("characteristic_zero", "特征零", r"\bcharacteristic\s+(?:zero|0)\b|特征(?:为)?零|特征(?:为)?0"),
)
_QUALIFIERS = tuple((key, label, re.compile(pattern, re.I | re.S)) for key, label, pattern in _QUALIFIERS)
_OBJECT_PATTERN = r"\$[^$\r\n]{1,160}\$|\\\([^\r\n]{1,160}?\\\)|[A-Za-z][A-Za-z0-9_'/\\-]{0,30}"
_LET_DECL = re.compile(r"\b(?:let|suppose|assume)\s+(?P<object>" + _OBJECT_PATTERN + r")\s+(?:be|is)\s+(?P<description>[^.;\r\n]{1,220})", re.I)
_FOR_DECL = re.compile(r"\bfor\s+(?:any|every|each|all)\s+(?P<description>[^$.;\r\n]{1,140}?)\s*(?P<object>\$[^$\r\n]{1,160}\$)", re.I)
_BASE_TYPES = re.compile(r"\b(?:integral\s+domain|domain|field\s+extension|ideal\s+sheaf|representation|ring|group|space|solution|vector|element|module|field|morphism|map|function|operator|ideal|functor|process|distribution|variety|scheme|subgroup|point|line|manifold|curve|surface)\b", re.I)


def _object_key(value: str) -> str:
    value = value.strip()
    if value.startswith("$") and value.endswith("$"):
        value = value[1:-1]
    elif value.startswith(r"\(") and value.endswith(r"\)"):
        value = value[2:-2]
    return _math_normalize(value)  # Keep mathematical case, subscripts and negation.


def _base_type(description: str) -> str:
    match = _BASE_TYPES.search(description)
    if not match:
        return "unknown"
    value = re.sub(r"\s+", " ", match.group(0).casefold())
    return "ring" if value in {"domain", "integral domain"} else value


def _description_properties(description: str, base: str) -> dict[str, str]:
    properties = {}
    for key, label, pattern in _QUALIFIERS:
        if key == "nonzero_alternative":
            continue
        match = pattern.search(description)
        if not match or (match.start() and description[match.start() - 1] == "\\"):
            continue
        if re.search(r"\b(?:not|no|without)\s+(?:necessarily\s+)?(?:an?\s+)?$", description[max(0, match.start() - 30):match.start()], re.I):
            continue
        if key == "nonzero" and re.search(r"\bat\s+least\s+one\b", description[:match.start()], re.I):
            continue  # An alternative over several objects is not each object's property.
        if key not in {"finite_dimension", "finite_generation", "finite_presentation", "nonzero"}:
            if ((match.start() and description[match.start() - 1] in "-‐‑") or
                    (match.end() < len(description) and description[match.end()] in "-‐‑")):
                continue
        if key == "normal" and base == "distribution" and re.search(r"\bGaussian\b", description, re.I):
            continue
        properties[key] = label
    if base == "ring" and re.search(r"\b(?:integral\s+)?domain\b", description, re.I):
        properties["integral_domain"] = "ring → domain/integral domain"
    if base=='manifold' and re.search(r'\bcomplex\s+manifold\b',description,re.I) and not re.search(r'\b(?:not|non)\b',description,re.I):
        properties['smooth']='光滑（复流形类型已包含；不把补写计作新增）'
    return properties


def _declarations(text: str) -> list[dict]:
    active = masks(text)[1]
    records, used = [], set()
    for pattern in (_LET_DECL, _FOR_DECL):
        for match in pattern.finditer(active):
            obj = _object_key(match.group("object"))
            if not obj or (match.start(), obj) in used:
                continue
            scope = _scope(active, match.start())
            # A postposed universal range ("P(a) for every nonzero ideal a")
            # restricts the quantified input even after the displayed guarantee.
            if scope == "conclusion_like_text" and pattern is not _FOR_DECL:
                continue
            description = match.group("description")
            base = _base_type(description)
            used.add((match.start(), obj))
            records.append({"object": obj, "object_quote": match.group("object"), "base": base,
                            "description": description, "properties": _description_properties(description, base),
                            "start": match.start(), "end": match.end(), "scope": "universal_domain_modifier_syntax" if pattern is _FOR_DECL else "declaration_premise_syntax"})
    return records


# Generic noun-phrase modifiers: "Let $G$ be a graph" -> "a planar graph".
# The head is the last word before the first qualifying connective; only words
# added in front of an unchanged head are reported.
_NP_CONNECTIVE = re.compile(r"\s(?:with|such\s+that|satisfying|of|whose|which|that|for|in|on|over|where|and|or|containing|having)\b|,", re.I)


_NP_WORD = re.compile(r"\$[^$]{1,40}\$(?:-[A-Za-z]+)*|[A-Za-z][A-Za-z'-]*")
_NP_IGNORED = frozenset("a an the any arbitrary some given fixed each every all certain".split())
_NP_NEGATED = re.compile(r"\b(?:not|no|without|necessarily)\b", re.I)


def _noun_phrase(description: str) -> tuple[str, list[str]] | None:
    cut = _NP_CONNECTIVE.search(description)
    phrase = description[:cut.start()] if cut else description
    if _NP_NEGATED.search(phrase):
        return None
    words = list(_NP_WORD.finditer(phrase))
    if not words or words[-1].group(0).startswith("$"):
        return None
    head = words[-1].group(0).casefold()
    modifiers = [m.group(0) for m in words[:-1] if m.group(0).casefold() not in _NP_IGNORED]
    return head, modifiers


def _hypothesis_clauses(text: str) -> list[Clause]:
    return clauses(text)


def _object_mention(text: str, key: str) -> tuple[int, int] | None:
    for unit in _math_units(text):
        if unit["normalized"] == key:
            return unit["start"], unit["end"]
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_']*", key):
        match = re.search(r"(?<![A-Za-z0-9_\\])" + re.escape(key) + r"(?![A-Za-z0-9_'])", _active_text(text))
        return (match.start(), match.end()) if match else None
    return None


def _simple_predicate(body: str) -> tuple[str, str] | None:
    """Only a positive, single-object adjectival predicate; never bag-of-words."""
    body = masks(body)[1]
    match = re.fullmatch(r"\s*(?P<object>" + _OBJECT_PATTERN + r")\s+is\s+([A-Za-z]+(?:-[A-Za-z]+)*)\s*", body)
    if not match or _NP_NEGATED.search(body):
        return None
    return _object_key(match.group('object')), match.group(2).casefold()


def _new_premises(old: str, new: str) -> list[dict]:
    a, b = _active_text(old), _active_text(new)
    old_decls, new_decls = _declarations(a), _declarations(b)
    old_predicates = {_simple_predicate(m.group('body')) for m in _hypothesis_clauses(a)}
    # Enumeration-local bindings require a scope parser; don't let these new
    # generic rules join declarations from separate branches.
    generic_scope_supported = not re.search(r"\\item\b|\\begin\s*\{(?:enumerate|cases)\}|\([ivx]+\)", a + b)
    findings = []

    def add(prop, label, obj, old_span, new_span, scope, phrase, base="unknown"):
        if any(p["property"] == prop and p["object_key"] == obj for p in findings):
            return
        old_text = a[old_span[0]:old_span[1]]
        new_text = b[new_span[0]:new_span[1]]
        identity = prop + "\0" + obj + "\0" + new_text
        findings.append({"premise_id": "prem_" + sha256(identity.encode("utf-8")).hexdigest()[:20],
            "property": prop, "label": label, "object_key": obj,
            "old_minimal": old_text, "new_minimal": new_text,
            "old_span": list(old_span), "new_span": list(new_span),
            "span_unit": "characters_within_extracted_statement", "scope": scope,
            "object_alignment": "same_case_sensitive_textual_object_not_mathematical_identity",
            "association_phrases": phrase, "base_type": base,
            "logical_strength": "unknown", "global_assumptions_checked": False})

    for nd in new_decls:
        matches = [od for od in old_decls if od["object"] == nd["object"] and od["base"] == nd["base"] and od["base"] != "unknown"]
        if not matches:
            continue
        od = matches[0]
        for prop in sorted(nd["properties"].keys() - od["properties"].keys()):
            property_pattern = next((p for key, _, p in _QUALIFIERS if key == prop), None)
            if generic_scope_supported and property_pattern and any(
                    predicate and predicate[0] == nd['object'] and property_pattern.fullmatch(predicate[1])
                    for predicate in old_predicates):
                continue
            phrase = []
            if prop == "integral_domain":
                phrase = ["integral domain", "local domain"]
            else:
                pattern = next((p for key, _, p in _QUALIFIERS if key == prop), None)
                match = pattern.search(nd["description"]) if pattern else None
                if match:
                    phrase = [match.group(0).replace("-", " ") + " " + nd["base"]]
            add(prop, "同一对象新增词面属性：" + nd["properties"][prop], nd["object"],
                (od["start"], od["end"]), (nd["start"], nd["end"]), nd["scope"], phrase, nd["base"])

    # Any adjective-like word added before the same head noun of the same object.
    for nd in new_decls:
        new_np = _noun_phrase(nd["description"])
        if not new_np or not generic_scope_supported or sum(d['object'] == nd['object'] for d in old_decls) != 1:
            continue
        for od in old_decls:
            old_np = _noun_phrase(od["description"]) if od["object"] == nd["object"] else None
            if not old_np or old_np[0] != new_np[0]:
                continue
            old_words = {w.casefold() for w in old_np[1]}
            if not old_words <= {w.casefold() for w in new_np[1]}:
                continue  # Replacement, not a supported monotone addition.
            for word in new_np[1]:
                if word.casefold() in old_words or (nd['object'], word.casefold()) in old_predicates:
                    continue
                known = [key for key, _, pattern in _QUALIFIERS if pattern.fullmatch(word.replace("-", " "))]
                if any(key in od["properties"] for key in known) or any(
                        p["object_key"] == nd["object"] and p["property"] in known for p in findings):
                    continue
                add("modifier:" + word.casefold(), "同一对象新增修饰词：" + word, nd["object"],
                    (od["start"], od["end"]), (nd["start"], nd["end"]), nd["scope"],
                    [word + " " + new_np[0]], new_np[0])
            break

    # New hypothesis clauses ("If $G$ is planar, then ...") before the conclusion.
    # Exact old clauses or a positive property of the same object can cover a
    # moved condition. Words belonging to another object never provide coverage.
    old_bodies = {body_key(m.group('body')) for m in _hypothesis_clauses(a)}
    existing = {tuple(row['new_span']) for row in comparison(a, b)}
    for match in _hypothesis_clauses(b):
        body = _normalized(match.group("body"))
        if (not generic_scope_supported or not body or body_key(match.group('body')) in old_bodies
                or (match.start(), match.end()) in existing):
            continue
        key = subject(match.group('body'))
        if key and any(subject(c.group('body')) == key for c in _hypothesis_clauses(a)):
            continue  # Multiple old predicates: unresolved correspondence, not absent.
        if any(d["start"] < match.end() and match.start() < d["end"] for d in new_decls):
            continue  # "Suppose $G$ is a planar graph" is compared as a declaration above.
        # Formula-only clauses ("If $n\\geq 3$") are left to formula_scope.
        words = _anchor_words(re.sub(r"\$[^$]*\$", " ", match.group("body")))
        if not words:
            continue
        predicate = _simple_predicate(match.group('body'))
        if predicate and (predicate in old_predicates or any(
                d['object'] == predicate[0] and (np := _noun_phrase(d['description']))
                and predicate[1] in {w.casefold() for w in np[1]} for d in old_decls)):
            continue
        units = _math_units(match.group("body"))
        obj = units[0]["normalized"] if units else ""
        om = _object_mention(a, obj) if obj else None
        if not om:
            continue  # No aligned object: preserve raw changes without a premise claim.
        # A bare object mention is insufficient old evidence. Require one
        # declaration of that object and retain its complete predicate.
        anchors = [d for d in old_decls if d['object'] == obj]
        if len(anchors) != 1:
            continue
        om = (anchors[0]['start'], anchors[0]['end'])
        add("hypothesis:" + body, "新增假设从句：" + match.group(0).strip(), obj,
            om or (0, 0), (match.start(), match.end()), "hypothesis_clause_syntax",
            [match.group("body").strip()], "hypothesis_clause")

    rational = re.compile(r"(?P<object>\$[^$\r\n]{1,120}\$)\s+(?:lying|lies|contained)\s+(?:on|in)\s+(?:some\s+|an?\s+)?rational[-\s]+line\b", re.I)
    for match in rational.finditer(b):
        obj = _object_key(match.group("object"))
        om = _object_mention(a, obj)
        already = any(_object_key(m.group("object")) == obj for m in rational.finditer(a))
        if om and not already and _scope(b, match.start()) != "conclusion_like_text":
            add("rational_line", "同一参数新增有理直线限定词面", obj, om, (match.start(), match.end()),
                "object_domain_modifier_syntax", ["rational line"], "point")

    dense = re.compile(r"(?:\bthe\s+)?\bimage\s+of\s+(?P<object>" + _OBJECT_PATTERN + r")\s+is\s+(?:a\s+)?dense(?:\s+subset)?(?:\s+(?:of|in))?[^.;\r\n]{0,160}|(?P<subject>" + _OBJECT_PATTERN + r")\s+has\s+(?:a\s+)?dense\s+image\b", re.I)
    old_dense = {_object_key(m.group("object") or m.group("subject")) for m in dense.finditer(a)}
    for match in dense.finditer(b):
        obj = _object_key(match.group("object") or match.group("subject"))
        om = _object_mention(a, obj)
        if om and obj not in old_dense and _scope(b, match.start()) != "conclusion_like_text":
            add("dense_image", "同一映射新增稠密像条件词面", obj, om, (match.start(), match.end()),
                "assumption_predicate_syntax", ["dense image", "dense subset"], "morphism")
    return findings


def _tightening_evidence(old: str, new: str, premises: list[dict], rewrite: dict) -> dict:
    ac, bc = _conclusion(old), _conclusion(new)
    same = bool(ac and bc and _normalized(ac) == _normalized(bc))
    af = {u["normalized"] for u in _math_units(old) if len(u["normalized"]) > 15}
    bf = {u["normalized"] for u in _math_units(new) if len(u["normalized"]) > 15}
    shared_guarantee = bool(af & bf and re.search(r"\bis\s+at\s+(?:least|most)\b|\bbounded\s+(?:above|below)\b", old, re.I)
                            and re.search(r"\bis\s+at\s+(?:least|most)\b|\bbounded\s+(?:above|below)\b", new, re.I))
    relation = {"status": "same_visible_conclusion_text" if same else "shared_guarantee_formula" if shared_guarantee else "mixed_or_unverified",
                "label": "可见结论词元相同（全局语义未核）" if same else "可见保证公式有共同锚点（并非完整结论认证）" if shared_guarantee else "结论或其他公式也变了，整体方向未定"}
    status = "rewrite_only" if rewrite["suppresses_strength_alert"] else "aligned_new_premises" if premises and (same or shared_guarantee) else "aligned_new_premises_with_other_changes" if premises else "no_aligned_new_premise_detected"
    return {"status": status, "label": "同对象新增前提词面，值得核查适用范围" if premises else "未检出满足对象对齐规则的新增前提；不等于没有变化",
            "premise_count": len(premises), "conclusion_match": relation, "logical_weakening": "unknown",
            "scope": "局部声明句法属性与对象匹配；未执行定义展开、全局前提核验或数学蕴含证明。"}
