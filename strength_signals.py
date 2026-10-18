"""Deterministic, bounded triage of possible changes in mathematical statements.

This module neither retrieves papers nor calls a model.  Its scores order an
investigation queue; they are not probabilities, mathematical entailment tests,
or a finding that an author silently changed an incorrect theorem.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from hashlib import sha256
from typing import Iterable
import re
import unicodedata
from asymptotic_order import compare_bounds
from formula_scope import added_formula_premises, analyze_formula_scope, classify_change, condition_rewrite_projection
from statement_text import MAX_CONTEXT, MAX_DISPLAY_TEXT, MAX_TEXT, MAX_TOKENS, _BEGIN_DECL, _CITATION, _ENV, _EXISTENTIAL, _MATH_BLOCK, _META, _NEGATION, _NUMBER, _PROOF, _RELATION, _STYLE, _TOKEN, _UNIVERSAL, _active_text, _formula_backreference, _formula_evidence, _formulas, _math_units, _normalized, _notation_projection, _paper_id, _quote, _scope, _semantic_active, _signal, _similarity, _string, _tokens, _version
from alignment import _add_competition, _alignment, _title
from premises import _QUALIFIERS, _new_premises, _tightening_evidence
from disclosure import _disclosure, _disclosure_documents

VERSION = "0.6.1"
__version__ = VERSION


_RESTRICTED_USE = re.compile(
    r"\b(?:special|particular)\s+case\b|\bwe\s+only\s+need\b|\bunder\s+(?:the\s+)?additional\s+assumptions?\b|"
    r"\brestrict(?:ed|ing)\s+to\b|\bwe\s+(?:use|apply)\b|这里只需|特例|仅使用|应用于", re.I)
_CITED_USE = re.compile(
    r"\bby\s+(?:the\s+)?(?:theorem|lemma|proposition|corollary)\b|\bapplying\b|"
    r"\b(?:theorem|lemma|proposition|corollary)\b.{0,70}?\\cite|由.{0,12}?(?:定理|引理|命题)", re.I | re.S)
_EQUIV_INTRO = re.compile(r"\b(?:the\s+)?following\s+(?:(?:statements?|conditions?|assertions?)\s+)?are\s+equivalent\b", re.I)
_INLINE_ITEM = re.compile(r"(?:\$\s*)?\(\s*(?:\{\s*\\rm\s+|\{\s*\\mathrm\s*\{?|\\mathrm\s*\{)?([ivxlcdm]+|\d+)\s*\}*\s*\)(?:\s*\$)?", re.I)


def _date(side: dict):
    text = _string(side.get("date")).strip()
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        precise = "T" in text or bool(re.search(r"\d\d:\d\d", text))
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value, precise
    except (ValueError, TypeError):
        return None


def _date_order(a, b) -> int:
    if not a or not b:
        return 0
    av, ap = a
    bv, bp = b
    if av.date() == bv.date() and not (ap and bp):
        return 0
    return (av > bv) - (av < bv)


def _snapshot_date(side: dict) -> bool:
    return _string(side.get("date_kind")).casefold() in {
        "version_date", "snapshot_date", "snapshot", "source_snapshot_date",
        "arxiv_version_date", "actual_version_date", "version_submission_date",
        "official_arxiv_version_history_utc",
    }


def _chronology(a: dict, b: dict) -> tuple[dict, bool]:
    same = bool(_paper_id(a)) and _paper_id(a) == _paper_id(b)
    va, vb = _version(a), _version(b)
    da, db = _date(a), _date(b)
    date_order = _date_order(da, db)
    both_snapshots = _snapshot_date(a) and _snapshot_date(b)
    result = {
        "status": "unknown", "direction": "unknown", "basis": [],
        "label": "先后待定", "statement_order_confirmed": False,
        "dates_are_statement_snapshot_dates": both_snapshots,
        "same_paper": same,
    }
    order = 0
    if same and va is not None and vb is not None and va != vb:
        order = (va > vb) - (va < vb)
        if both_snapshots and date_order and date_order != order:
            result.update(status="conflict", label="版本序号与实际版本日期冲突，方向待核", basis=["version_number", "conflicting_version_dates"])
            return result, False
        result.update(status="version_order", label="同篇版本序号可排序；未据此补造版本日期", basis=["same_paper_version_number"], statement_order_confirmed=True)
    elif same and va is not None and vb is not None and va == vb:
        result.update(label="同一版本，不能据输入顺序建立修订方向", basis=["same_version"])
        return result, False
    elif both_snapshots and date_order:
        order = date_order
        result.update(status="version_date_order", label="按所给实际版本日期排序", basis=["provided_version_dates"], statement_order_confirmed=True)
    elif not same and da and db and date_order:
        # Paper-level dates may rank papers, but not the statements in unknown
        # later source snapshots.  In particular first_submission is not a
        # statement timestamp.
        order = date_order
        result.update(status="paper_date_only", label="仅有论文日期先后；语句快照先后未确认", basis=["provided_paper_dates_only"])
    elif da and db and da[0].date() == db[0].date():
        result.update(label="同日且缺少可区分的实际版本信息，方向待定", basis=["same_day_without_distinct_version"])
    else:
        result.update(basis=["missing_or_noncomparable_version_dates"])
    if order:
        result["direction"] = "reversed" if order > 0 else "as_given"
    return result, order > 0


def _name(name: str) -> str:
    # Exact normalized spelling only.  Initials, surname variants, transliteration,
    # and shared names are deliberately not turned into verified people.
    value = unicodedata.normalize("NFKC", _string(name)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", value).split())


def _names(side: dict) -> dict[str, str]:
    values = side.get("authors", [])
    if not isinstance(values, list):
        values = []
    return {_name(value): _string(value) for value in values if _name(value)}


def _author_relation(old: dict, new: dict, hint) -> dict:
    a, b = _names(old), _names(new)
    shared = sorted(a.keys() & b.keys())
    if _paper_id(old) and _paper_id(old) == _paper_id(new):
        kind, label = "same_paper", "同篇不同或未明版本"
    elif shared:
        kind, label = "name_overlap", "作者姓名有重合（未经人物消歧）"
    elif a and b:
        kind, label = "no_name_overlap", "作者姓名未重合（第三方候选，身份未核）"
    else:
        kind, label = "unknown", "作者元数据不全，关系待核"
    return {
        "kind": kind, "label": label, "matched_names": [a[key] for key in shared],
        "identity_verified": False, "basis": "normalized_corpus_metadata_name_spelling_only_not_version_author_rosters",
        "historical_version_author_lists_verified": False,
        "metadata_scope_note": "姓名来自所给论文元数据，未人物消歧，可能不是两端版本各自的历史作者名单。",
        "author_match_hint": _string(hint)[:400] if hint else None,
        "hint_used_for_identity": False,
    }


def _statement(side: dict) -> tuple[str, dict, list[str], bool]:
    raw = _string(side.get("text"))
    source = side.get("source") if isinstance(side.get("source"), dict) else {}
    rawscan = raw[:MAX_TEXT]
    scan = _active_text(rawscan)
    start, end, reasons = 0, len(scan), []
    first = _BEGIN_DECL.search(scan)
    if first:
        prefix = _META.sub("", scan[:first.start()]).strip()
        if not prefix:
            start = first.start()
            # A proof or another declaration can occur before the closing env in
            # a polluted extraction. Neither belongs to the first statement.
            closing = re.search(r"\\end\s*\{" + re.escape(first.group(1)) + r"\*?\}", scan[first.end():], re.I)
            if closing:
                end = min(end, first.end() + closing.end())
            next_decl = _BEGIN_DECL.search(scan, first.end())
            if next_decl and next_decl.start() < end:
                end = next_decl.start()
                reasons.append("before_next_declaration")
        else:
            end = min(end, first.start())
            reasons.append("before_appended_declaration")
    proof = _PROOF.search(scan, start)
    if proof and proof.start() < end:
        end = proof.start()
        reasons.append("before_proof")
    segment = rawscan[start:end]
    leading, trailing = len(segment) - len(segment.lstrip()), len(segment) - len(segment.rstrip())
    start += leading
    end = max(start, end - trailing)
    text = rawscan[start:end]
    warnings = []
    if reasons:
        warnings.append("声明提取在证明或后续声明之前截断；后文不参与本命题变化判定。")
    if len(raw) > MAX_TEXT or source.get("input_text_truncated") or source.get("preview_may_be_truncated"):
        warnings.append("输入声明或预览可能截断；缺失尾部不能解释为数学条件被删除。")
    if source.get("post_first_end_document"):
        warnings.append("源片段位于首个 end{document} 之后，所属有效文档未确定；未将整段自动判假。")
    roots = source.get("document_roots")
    if isinstance(roots, (int, float)) and roots > 1:
        warnings.append("源文件含多个文档根，片段归属与有效版本需要核对。")
    if source.get("source_available") is False or not source.get("path"):
        warnings.append("原文来源不可用或路径缺失，当前只能核对所给文本。")
    if not _version(side):
        warnings.append("实际源版本号未知，不能把首次上传日期当作该语句的版本日期。")
    if not _string(side.get("context")):
        warnings.append("局部上下文缺失；全局假设、引用角色和更正说明可能未覆盖。")
    elif len(_string(side.get("context"))) > MAX_CONTEXT:
        warnings.append("局部上下文超过分析上限，仅检查有界窗口。")
    commented = bool(re.search(r"\\begin\s*\{comment\}", rawscan))
    nonblank = [line.lstrip() for line in rawscan.splitlines() if line.strip()]
    if nonblank and sum(line.startswith("%") for line in nonblank) / len(nonblank) > 0.5:
        commented = True
    if commented:
        warnings.append("片段可能处于注释或停用区域，不能默认是有效定理。")
    serious = bool(source.get("post_first_end_document") or (isinstance(roots, (int, float)) and roots > 1) or commented)
    derivation = {
        "start": start, "end": end, "unit": "characters_within_input_text",
        "boundary_rules": reasons or ["provided_statement_window"],
        "input_text_length": len(raw), "analysis_window_limit": MAX_TEXT,
        "source_span_is_original_input_span": True,
        "returned_statement_characters": len(text),
        "coordinate_note": "start/end是所给输入text的字符坐标，已计入首尾裁剪；source.span有独立来源/单位，未做源码坐标转换。",
    }
    return text, derivation, warnings, serious


def _qualifier_signals(old: str, new: str) -> list[dict]:
    a, b = _semantic_active(old), _semantic_active(new)
    signals = []

    def occurrences(text, key, pattern):
        found = {}
        for match in pattern.finditer(text):
            # A command named \\smooth is notation, not the adjective smooth.
            if match.start() and text[match.start() - 1] == "\\":
                continue
            compound = None
            if key not in {"finite_dimension", "finite_generation", "finite_presentation", "nonzero_alternative", "nonzero"}:
                left, right = match.start(), match.end()
                while left > 0 and re.match(r"[A-Za-z\-‐‑]", text[left - 1]):
                    left -= 1
                while right < len(text) and re.match(r"[A-Za-z\-‐‑]", text[right]):
                    right += 1
                value = text[left:right]
                if re.search(r"[\-‐‑]", value):
                    compound = value
                if key == "finite":
                    phrase = re.match(r"finite\s+limit\s+preserving\b", text[match.start():], re.I)
                    if phrase:
                        left, right = match.start(), match.start() + phrase.end()
                        compound = text[left:right]
                if compound:
                    found[("compound", compound.casefold())] = (left, right, compound)
                    continue
            found[("qualifier", key)] = (match.start(), match.end(), match.group(0))
        return found

    for key, label, pattern in _QUALIFIERS:
        am, bm = occurrences(a, key, pattern), occurrences(b, key, pattern)
        for occurrence_key in sorted(am.keys() ^ bm.keys()):
            added = occurrence_key in bm
            target = b if added else a
            pos, end, value = (bm if added else am)[occurrence_key]
            scope = _scope(target, pos)
            evidence = _formula_evidence(new if added else old, {"start": pos, "end": end, "environment": "dollar_inline"})
            quote = evidence["quote"]
            compound = occurrence_key[0] == "compound"
            if compound:
                direction = "unresolved_strength_direction"
                explanation = "保留完整复合术语；其中的词不能单独解释为对象范围的新前提。当前未解析该术语的定义、前提或结论角色。"
                kind, shown_label = "terminology_change", "复合术语文字变化：" + value
                scope = "compound_terminology_role_unknown"
            else:
                direction = ("possible_premise_tightening" if added else "possible_premise_relaxation") if scope == "premise_like_text" else "unresolved_strength_direction"
                explanation = "第二端新增该词面限定。" if added else "第一端词面限定在第二端未见。"
                explanation += "前提/结论角色只有句法提示；可能是条件、结论、对象名称、换述或特例应用，不能仅据该词认证逻辑强弱。"
                kind, shown_label = "qualifier_change", ("新增限定词面：" if added else "未再出现限定词面：") + label
            signals.append(_signal(
                kind, shown_label, _quote(a) if added else quote, quote if added else _quote(b),
                explanation, scope, term=value.casefold() if compound else key,
                action="added" if added else "removed", direction_hint=direction,
                logical_role_status="unknown", syntactic_role_hint=scope,
                quote_source="comment_masked_active_statement",
                changed_side="new" if added else "old", changed_span=[pos, end],
                changed_context_span=evidence["span"], context_truncated=evidence["context_truncated"],
            ))
    return signals


_ASYMPTOTIC_LABELS = {
    ("upper", "weaker"): "明确极限下的结论上界放宽",
    ("upper", "stronger"): "明确极限下的结论上界收紧",
    ("lower", "weaker"): "明确极限下的结论下界放宽",
    ("lower", "stronger"): "明确极限下的结论下界收紧",
}


def _asymptotic_signals(old: str, new: str) -> list[dict]:
    signals = []
    for f in compare_bounds(_active_text(old), _active_text(new)):
        label = _ASYMPTOTIC_LABELS.get((f["kind"], f["direction"]), "结论中的 Θ 界增长阶改变（方向未定）")
        signals.append(_signal(
            "asymptotic_bound_change", label, f["old_quote"], f["new_quote"],
            "两端明确极限：" + ("变量趋于零" if f['limit'] == 'zero' else "变量趋于正无穷") + "。同一主语、相同渐近记号的局部比较；趋于零时反转幂的大小次序且不比较对数因子。未认证整条命题的蕴含方向。",
            f["role"], action=f["direction"], direction_hint="local_conclusion_bound_" + f["direction"],
            old_span=f["old_span"], new_span=f["new_span"], span_unit="characters_within_extracted_statement",
            asymptotic_subject=f["subject"], asymptotic_limit=f['limit'], old_order=f["old_order"], new_order=f["new_order"],
            logical_role_status="bounded_syntax_not_global_entailment"))
    return signals


def _polarity_signals(old: str, new: str) -> list[dict]:
    a, b = _semantic_active(old), _semantic_active(new)
    signals = []
    au, bu = list(_UNIVERSAL.finditer(a)), list(_UNIVERSAL.finditer(b))
    ae, be = list(_EXISTENTIAL.finditer(a)), list(_EXISTENTIAL.finditer(b))
    if (bool(au) != bool(bu)) or (bool(ae) != bool(be)):
        aq = _UNIVERSAL.search(a) or _EXISTENTIAL.search(a)
        bq = _UNIVERSAL.search(b) or _EXISTENTIAL.search(b)
        action = "universal_to_existential" if au and not bu and be else "existential_to_universal" if bu and not au and ae else "quantifier_presence_change"
        signals.append(_signal(
            "quantifier_change", "全称或存在量词的文字形式变化",
            _quote(a, aq.start(), aq.end()) if aq else _quote(a),
            _quote(b, bq.start(), bq.end()) if bq else _quote(b),
            "保留了全称、存在量词；前提中的量词与结论中的量词作用不同，当前未完成变量绑定与蕴含判定。",
            action=action, direction_hint="unresolved_strength_direction",
            old_universal_count=len(au), new_universal_count=len(bu),
            old_existential_count=len(ae), new_existential_count=len(be),
            logical_role_status="unknown", quote_source="comment_masked_active_statement",
            old_span=[aq.start(), aq.end()] if aq else None,
            new_span=[bq.start(), bq.end()] if bq else None,
            span_unit="characters_within_extracted_statement",
        ))
    an, bn = list(_NEGATION.finditer(a)), list(_NEGATION.finditer(b))
    if len(an) != len(bn):
        aq, bq = _NEGATION.search(a), _NEGATION.search(b)
        dependency = bool((aq and re.match(r"not\s+(?:necessarily\s+)?depend\w*\b", a[aq.start():], re.I)) or
                          (bq and re.match(r"not\s+(?:necessarily\s+)?depend\w*\b", b[bq.start():], re.I)))
        signals.append(_signal(
            "dependency_phrase_change" if dependency else "negation_change",
            "依赖关系措辞的否定变化" if dependency else "否定词或不等符号文字出现次数变化（作用域未知）",
            _quote(a, aq.start(), aq.end()) if aq else _quote(a),
            _quote(b, bq.start(), bq.end()) if bq else _quote(b),
            "否定与不等符号未被当作停用词删除。出现次数变化是核查线索，尚未判定否定作用域或两个命题的逻辑关系。",
            action="negation_count_changed", old_count=len(an), new_count=len(bn), direction_hint="unresolved_strength_direction",
            logical_role_status="unknown", quote_source="comment_masked_active_statement",
            negation_scope_status="unknown",
            old_span=[aq.start(), aq.end()] if aq else None,
            new_span=[bq.start(), bq.end()] if bq else None,
            span_unit="characters_within_extracted_statement",
        ))
    return signals


def _formula_signals(old: str, new: str, similarity: float) -> list[dict]:
    old, new = _active_text(old), _active_text(new)
    af, bf = _math_units(old), _math_units(new)
    aset, bset = {x["normalized"] for x in af}, {x["normalized"] for x in bf}
    removed = [x for x in af if x["normalized"] not in bset]
    added = [x for x in bf if x["normalized"] not in aset]
    notation_pairs = []
    notation_a, notation_b = set(), set()
    for a in removed:
        for b in added:
            if a["start"] in notation_a or b["start"] in notation_b:
                continue
            if _notation_projection(a["normalized"]) == _notation_projection(b["normalized"]):
                notation_pairs.append((a, b))
                notation_a.add(a["start"])
                notation_b.add(b["start"])
    removed = [a for a in removed if a["start"] not in notation_a]
    added = [b for b in added if b["start"] not in notation_b]
    # Include an old unit still present in an unchanged new branch as an anchor
    # for a different branch. A one-to-one subtraction would miss that revision.
    new_objects = {u["object"] for u in added if u["object"]}
    old_objects = {u["object"] for u in removed if u["object"]}
    a_candidates = [u for u in af if u in removed or u["object"] in new_objects]
    b_candidates = [u for u in bf if u in added or u["object"] in old_objects]
    matches = []
    for a in a_candidates[:30]:
        at = set(_TOKEN.findall(a["normalized"]))
        for b in b_candidates[:30]:
            if a["normalized"] == b["normalized"]:
                continue
            if not a["object"] or not b["object"] or a["object"] != b["object"]:
                continue
            bt = set(_TOKEN.findall(b["normalized"]))
            overlap = len(at & bt) / max(1, len(at | bt))
            same_numeric_skeleton = _NUMBER.sub("#", a["normalized"]) == _NUMBER.sub("#", b["normalized"])
            relation_weight = .20 if a["relation"] and b["relation"] else 0
            object_weight = .5 if a["object"] and a["object"] == b["object"] else 0
            quality = overlap + (0.5 if same_numeric_skeleton else 0) + relation_weight + object_weight
            if overlap >= .20:
                matches.append((quality, a, b))
    if similarity < .20:
        return []
    signals, used_a, used_b = [], set(), set()
    for _, a, b in sorted(matches, key=lambda x: (-x[0], x[1]["start"], x[2]["start"])):
        if b["start"] in used_b:
            continue
        # A single older expression can legitimately anchor several new branches.
        used_a.add(a["start"])
        used_b.add(b["start"])
        numbers_changed = _NUMBER.findall(a["normalized"]) != _NUMBER.findall(b["normalized"])
        relation_changed = _RELATION.findall(a["normalized"]) != _RELATION.findall(b["normalized"])
        kind = "inequality_change" if relation_changed else "numeric_formula_change" if numbers_changed else "formula_change"
        label = "等号、不等号或否定关系式文字变化" if relation_changed else "公式中的数值片段变化" if numbers_changed else "公式对象或表达式片段变化"
        ae, be = _formula_evidence(old, a), _formula_evidence(new, b)
        signals.append(_signal(
            kind, label, ae["quote"], be["quote"],
            "公式行的等号/不等号左端文字相同，据此形成文字候选配对。只归一化一份有限排版清单；仍可能有宏或记号换述，数学对象身份、作用域、界的大小及逻辑强度未认证。",
            "active_formula_text", action=kind, direction_hint="unresolved_strength_direction",
            logical_role_status="unknown", quote_source="comment_masked_active_statement",
            old_span=ae["span"], new_span=be["span"], span_unit="characters_within_extracted_statement",
            old_formula_span=ae["formula_span"], new_formula_span=be["formula_span"],
            old_formula_quote=ae["formula_quote"], new_formula_quote=be["formula_quote"],
            old_context_quote=ae["context_quote"], new_context_quote=be["context_quote"],
            old_context_span=ae["context_span"], new_context_span=be["context_span"],
            context_truncated=ae["context_truncated"] or be["context_truncated"],
            pairing_status="same_textual_left_hand_side", mathematical_object_identity_certified=False,
            old_comparison_object=a["object"], new_comparison_object=b["object"],
            old_branch_condition=a["condition"], new_branch_condition=b["condition"],
            new_comparison_object_inherited=b["inherited_object"],
            formula_change_structure="single_to_branches" if a["row_count"] == 1 and b["row_count"] > 1 else "branches_to_single" if a["row_count"] > 1 and b["row_count"] == 1 else "row_to_row",
            change_signature=sha256((a["normalized"] + "\0" + b["normalized"]).encode("utf-8")).hexdigest()[:20],
        ))
        if len(signals) >= 2:
            break
    if not signals:
        scope_pairs = [(a, b) for a in af for b in bf
                       if a["normalized"] == b["normalized"] and
                       _normalized(a["condition"]) != _normalized(b["condition"])]
        if scope_pairs:
            a, b = scope_pairs[0]
            signals.append(_signal(
                "branch_scope_change", "公式核心相同，分支适用范围文字发生变化",
                old[a["start"]:a["end"]], new[b["start"]:b["end"]],
                "同一公式核心在两端附带的分支条件不同；可能补写、划分或收缩参数范围，也可能明确既有全局约定，逻辑作用未认证。",
                "active_formula_branch_condition", action="branch_scope_text_changed",
                direction_hint="unresolved_strength_direction", logical_role_status="unknown",
                quote_source="comment_masked_active_statement",
                old_span=[a["start"], a["end"]], new_span=[b["start"], b["end"]],
                span_unit="characters_within_extracted_statement",
                old_comparison_object=a["object"], new_comparison_object=b["object"],
                old_branch_condition=a["condition"], new_branch_condition=b["condition"],
                change_signature=sha256((a["normalized"] + "\0" + _normalized(a["condition"]) + "\0" + _normalized(b["condition"])).encode("utf-8")).hexdigest()[:20],
            ))
    # Added/omitted formulas include newly written parameter domains. They are
    # observations, not a claim that a new assumption was mathematically needed.
    def conditional_first(units, source_text):
        def key(unit):
            evidence = _formula_evidence(source_text, unit)
            context = evidence["context_quote"]
            position = unit["start"] - evidence["context_span"][0]
            conditional = bool(unit["condition"] or (
                re.search(r"\bif\b", context[:max(0, position)], re.I) and
                re.search(r"\bthen\b", context, re.I)))
            return (not conditional, unit["start"])
        return sorted(units, key=key)

    unpaired_new = conditional_first([u for u in added if u["start"] not in used_b], new)
    unpaired_old = conditional_first([u for u in removed if u["start"] not in used_a], old)
    unmatched = []
    for index in range(2):
        if index < len(unpaired_new):
            unmatched.append(("added", unpaired_new[index]))
        if index < len(unpaired_old):
            unmatched.append(("omitted", unpaired_old[index]))
    for action, u in unmatched:
        kind = "formula_added" if action == "added" else "formula_omitted"
        source_text = new if action == "added" else old
        evidence = _formula_evidence(source_text, u)
        quote = evidence["quote"]
        backreference = _formula_backreference(source_text, old if action == "added" else new, u)
        parameter_domain = bool(_RELATION.search(u["normalized"]) or re.search(r"\\(?:in|notin|subset)(?![A-Za-z])", u["normalized"]))
        signals.append(_signal(
            kind, ("第二端新增或未配对的" if action == "added" else "第一端未再出现或未配对的") + ("参数范围/关系式片段" if parameter_domain else "公式片段"),
            backreference["quote"] if action == "added" and backreference else "" if action == "added" else quote,
            quote if action == "added" else backreference["quote"] if backreference else "",
            "有限排版归一化后仍未配对的活跃公式片段，不与另一端的独立对象硬配。可能是参数补写、宏/记号换述、回指、搬移或重述；不是假设实际新增/删去或逻辑强弱的认证。",
            "active_formula_text", action=action, direction_hint="unresolved_strength_direction",
            logical_role_status="unknown", quote_source="comment_masked_active_statement",
            changed_span=evidence["span"], changed_side="new" if action == "added" else "old",
            span_unit="characters_within_extracted_statement", parameter_domain_candidate=parameter_domain,
            formula_span=evidence["formula_span"], formula_quote=evidence["formula_quote"],
            context_quote=evidence["context_quote"], context_span=evidence["context_span"],
            context_truncated=evidence["context_truncated"], pairing_status="unpaired_fragment",
            mathematical_object_identity_certified=False,
            label_backreference=backreference,
            change_signature=sha256((action + "\0" + u["normalized"]).encode("utf-8")).hexdigest()[:20],
        ))
    for a, b in notation_pairs[:2]:
        ae, be = _formula_evidence(old, a), _formula_evidence(new, b)
        signals.append(_signal(
            "notation_change", "字体或表示法包装改写（数学作用未知）", ae["quote"], be["quote"],
            "去掉已列出的字体包装后文字相同；这只是表示法观察。粗体、正体或宏可能承担数学记号含义，因此没有据此认证纯排版或数学等价。",
            "active_formula_notation", action="font_or_notation_wrapper_changed",
            direction_hint="unresolved_strength_direction", logical_role_status="unknown",
            quote_source="comment_masked_active_statement", old_span=ae["span"], new_span=be["span"],
            old_formula_span=ae["formula_span"], new_formula_span=be["formula_span"],
            span_unit="characters_within_extracted_statement", pairing_status="notation_projection_only",
            change_signature=sha256((a["normalized"] + "\0" + b["normalized"]).encode("utf-8")).hexdigest()[:20],
        ))
    return signals


def _text_difference(old: str, new: str) -> dict:
    a, b = _semantic_active(old), _semantic_active(new)
    am, bm = list(_TOKEN.finditer(a))[:MAX_TOKENS], list(_TOKEN.finditer(b))[:MAX_TOKENS]
    av = [m.group(0).casefold() for m in am]
    bv = [m.group(0).casefold() for m in bm]
    af, bf = _formulas(old), _formulas(new)
    math_changed = [v[0] for v in af] != [v[0] for v in bf]
    coordinates = {"unit": "unicode_codepoints", "base": "extracted_statement_before_display_truncation",
                   "to_input_text": "add corresponding statement_slice.start",
                   "to_source": "not_converted_source_span_has_independent_basis"}
    if av == bv and not math_changed:
        return {"status": "identical" if old == new else "formatting_only", "changes": [],
                "label": "活跃正文的检测词元未见差异" if old == new else "仅检出排版、注释、标签或已忽略标点差异",
                "logical_strength": "unknown", "comparison_scope": "bounded_active_statement_tokens_not_full_paper",
                "coordinate_system": coordinates}
    projected_a, projected_b = _notation_projection(a), _notation_projection(b)
    only_notation = bool(
        (projected_a != a or projected_b != b) and _normalized(projected_a) == _normalized(projected_b)
        and [_notation_projection(v[0]) for v in af] == [_notation_projection(v[0]) for v in bf])
    changes = []
    # autojunk only affects matching anchors; it does not delete any negation or
    # inequality token from the sequences or the emitted changed spans.
    matcher = SequenceMatcher(None, av, bv, autojunk=True)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        aa = am[i1].start() if i1 < len(am) else len(a)
        az = am[i2 - 1].end() if i2 > i1 else aa
        ba = bm[j1].start() if j1 < len(bm) else len(b)
        bz = bm[j2 - 1].end() if j2 > j1 else ba
        changes.append({
            "kind": tag, "old_quote": _quote(a, aa, az, radius=60),
            "new_quote": _quote(b, ba, bz, radius=60),
            "old_changed_text": a[aa:az][:500], "new_changed_text": b[ba:bz][:500],
            "old_changed_text_truncated": az - aa > 500, "new_changed_text_truncated": bz - ba > 500,
            "old_changed_text_total_characters": az - aa, "new_changed_text_total_characters": bz - ba,
            "old_span": [aa, az], "new_span": [ba, bz],
            "span_unit": "characters_within_extracted_statement",
            "quote_source": "comment_masked_active_statement",
        })
        if len(changes) >= 8:
            break
    if not changes and math_changed:
        aset, bset = {x[0] for x in af}, {x[0] for x in bf}
        aa = next((x for x in af if x[0] not in bset), af[0] if af else None)
        bb = next((x for x in bf if x[0] not in aset), bf[0] if bf else None)
        changes.append({"kind": "formula_punctuation_or_structure_change",
                        "old_quote": a[aa[1]:aa[2]] if aa else "",
                        "new_quote": b[bb[1]:bb[2]] if bb else "",
                        "old_span": [aa[1], aa[2]] if aa else None,
                        "new_span": [bb[1], bb[2]] if bb else None,
                        "old_changed_text": a[aa[1]:aa[2]][:500] if aa else "",
                        "new_changed_text": b[bb[1]:bb[2]][:500] if bb else "",
                        "old_changed_text_truncated": bool(aa and aa[2] - aa[1] > 500),
                        "new_changed_text_truncated": bool(bb and bb[2] - bb[1] > 500),
                        "old_changed_text_total_characters": aa[2] - aa[1] if aa else 0,
                        "new_changed_text_total_characters": bb[2] - bb[1] if bb else 0,
                        "span_unit": "characters_within_extracted_statement",
                        "quote_source": "comment_masked_active_statement"})
    return {"status": "notation_only" if only_notation else "active_text_changed",
            "label": "只检出字体/表示法包装换述；数学作用未知" if only_notation else "活跃正文有变化；强度意义尚未判定", "changes": changes,
            "logical_strength": "unknown", "changes_display_limit": 8,
            "comparison_scope": "bounded_active_statement_tokens_not_full_paper", "coordinate_system": coordinates}


def _assertion_role(side: dict, text: str) -> dict:
    active = _active_text(text)
    citations = list(_CITATION.finditer(active))
    keys = sorted({key.strip() for match in citations for key in match.group(1).split(",") if key.strip()})
    first_clause = re.search(r"\b(?:let|suppose|assume|assuming|if|for)\b", active, re.I)

    def citation_location(position):
        for footnote in re.finditer(r"\\footnote(?:text)?(?:\[[^\]]*\])?\s*\{", active[:position], re.I):
            depth = 1
            for i in range(footnote.end(), position):
                if i and active[i - 1] == "\\":
                    continue
                depth += active[i] == "{"
                depth -= active[i] == "}"
            if depth > 0:
                return "inside_footnote"
        if first_clause and position < first_clause.start():
            return "statement_opening"
        return "statement_body_scope_unknown"

    locations = [citation_location(m.start()) for m in citations]
    evidence = [{"quote": _quote(active, m.start(), m.end()), "scope": "active_statement", "span": [m.start(), m.end()],
                 "visible_location": location, "attribution_scope": "unknown"}
                for m, location in list(zip(citations, locations))[:3]]
    source = side.get("source") if isinstance(side.get("source"), dict) else {}
    provided_kind = _string(source.get("declaration_kind") or side.get("declaration_kind") or side.get("kind")).strip().casefold().rstrip("*")
    environment = _BEGIN_DECL.search(active)
    environment_kind = environment.group(1).casefold() if environment else ""
    title = _title(active)
    opening = _ENV.sub("", _META.sub("", active)).lstrip()
    conjecture_header = bool(re.match(r"(?:conjecture\b|猜想|we\s+conjecture\s+that\b|it\s+is\s+conjectured\s+that\b)", opening, re.I))
    conjecture_title = bool(re.match(r"(?:conjecture\b|猜想)", title, re.I))
    if provided_kind in {"conjecture", "conj", "猜想"} or environment_kind in {"conjecture", "conj"} or conjecture_header or conjecture_title:
        role, label = "conjecture_cue", "含猜想角色线索，不能默认是已证明定理"
    elif provided_kind in {"definition", "def", "defn", "定义"} or environment_kind in {"definition", "defn"} or re.match(r"(?:definition\b|定义)", opening, re.I):
        role, label = "definition_cue", "定义类声明；强度含义需另核"
    elif citations:
        role, label = "citation_in_statement", "声明内有引用，作用范围未知；未据此认定整条定理来自他人"
    elif re.search(r"\b(?:due\s+to|proved\s+by|shown\s+by)\b|由.{0,20}?证明", active, re.I):
        role, label = "external_attribution_cue", "声明含外部归属措辞；具体原作者待核"
    else:
        role, label = "unresolved_assertion", "声明归属未知；未把论文作者自动认作命题原创作者"
    return {
        "role": role, "label": label, "ownership_status": "unknown",
        "citation_keys": keys[:15], "matched_quotes": evidence,
        "citation_locations": sorted(set(locations)), "citation_scope_status": "unknown",
        "declaration_kind_provided": provided_kind or None, "visible_declaration_environment": environment_kind or None,
        "role_basis": "declaration_metadata_or_environment_or_explicit_opening_header_not_body_topic_words",
        "opening_attribution_cue": "statement_opening" in locations,
        "scope": "provided_active_statement_only", "original_author_identity_verified": False,
        "paper_authors_are_not_assertion_owners_by_default": True,
    }


def _usage(new: dict, text: str) -> list[dict]:
    window = _active_text(text + "\n" + _string(new.get("context"))[:MAX_CONTEXT])
    match = _RESTRICTED_USE.search(window)
    kind = "restricted_use_context"
    if match is None:
        match = _CITED_USE.search(window)
        kind = "citation_use_context"
    if match is None:
        return []
    return [_signal(
        kind, "局部窗口含特例或应用措辞（目标命题未建立）" if kind == "restricted_use_context" else "局部窗口含引用或应用措辞（目标命题未建立）",
        "", _quote(window, match.start(), match.end()),
        "应用或特例不等于原命题改动；该线索仍保留，并可与本批其他后文聚合。当前未自动确认所引用对象就是第一端的命题。",
        "new_statement_and_local_context", action=kind, direction_hint="usage_pattern_not_logical_weakening",
        target_statement_adoption_status="unknown",
    )]


def _display_side(side: dict, text: str, derivation: dict) -> dict:
    value = {key: side[key] for key in (
        "paper_id", "statement_id", "version", "date", "date_kind", "first_submission_date",
        "authors", "author_identity", "source", "date_source",
        "statement_reference_keys", "statement_reference_basis") if key in side}
    value.update(text=text[:MAX_DISPLAY_TEXT], context=_string(side.get("context"))[:MAX_CONTEXT],
                 statement_slice=derivation, input_text_length=derivation["input_text_length"],
                 text_derivation="original_substring_bounded_to_first_statement",
                 display_text_truncated=len(text) > MAX_DISPLAY_TEXT,
                 statement_coordinates={"unit": "unicode_codepoints", "origin": "extracted_statement",
                    "input_start": derivation["start"], "input_end": derivation["end"],
                    "analysis_characters": len(text), "displayed_characters": min(len(text), MAX_DISPLAY_TEXT),
                    "source_coordinate_conversion": "not_performed"})
    return value


def _score(record: dict, serious: bool, missing: bool) -> None:
    algo = record["algorithm"]
    signals = algo["signals"]
    strength = [s for s in signals if s["kind"] not in {"restricted_use_context", "citation_use_context", "notation_change"}]
    components = []

    def add(key, points, reason):
        components.append({"key": key, "points": points, "reason": reason})

    alignment = algo["alignment_evidence"]
    if not strength or missing or algo["text_difference"]["status"] != "active_text_changed":
        add("no_strength_signal", 0, "当前未检出有效正文中的命名变化线索、仅排版差异或声明缺失；不表示已证明没有弱化。")
        score = 0
    elif {s["kind"] for s in strength} <= {"unclassified_text_change"}:
        add("unclassified_active_text_change", 8, "活跃正文确有变化，但未落入当前限定词/量词/公式规则；保留待核，未把无命名信号等同于没有数学变化。")
        score = 8
    else:
        sim = algo["lexical_similarity"]
        alignment_points = 12 if sim >= .75 else 9 if sim >= .5 else 5 if sim >= .3 else 2 if sim >= .15 else 0
        add("text_alignment", alignment_points, "文字重合仅支持候选对齐，不支持逻辑弱化；词袋相似度为 %.3f。" % sim)
        relation = algo["author_relation"]["kind"]
        add("author_or_version_relation", {"same_paper": 24, "name_overlap": 17, "no_name_overlap": 7, "unknown": 4}[relation],
            {"same_paper": "同篇版本文本候选优先核查；这不是命题原创归属认证。", "name_overlap": "论文作者姓名重合提高调查优先级；人物与语句归属均未核。", "no_name_overlap": "第三方文字变化保留为线索，优先级低于作者自身版本。", "unknown": "作者资料不全，保持候选并提示未知。"}[relation])
        roles = algo["assertion_role"]
        external_roles = {"citation_in_statement", "external_attribution_cue"}
        if relation in {"same_paper", "name_overlap"} and roles["old"]["role"] in external_roles:
            opening = roles["old"].get("opening_attribution_cue") or roles["old"]["role"] == "external_attribution_cue"
            add("assertion_ownership_unknown", -8 if opening else -2,
                "第一端开头有引用/归属线索，不能将论文作者重合当成原创归属。" if opening else "第一端正文/脚注有引用，作用范围未知，未据此判整条定理来自他人。")
        if any(v["role"] == "conjecture_cue" for v in roles.values()):
            add("conjecture_role", -10, "存在猜想角色线索，不能直接按已证明定理的修订理解。")
        qualifier = [s for s in strength if s["kind"] == "qualifier_change"]
        kinds = {s["kind"] for s in strength}
        raw_change_points = 0
        if qualifier:
            raw_change_points += 18 if any(s["action"] == "added" and s["scope"] == "premise_like_text" for s in qualifier) else 10 if any(s["scope"] == "premise_like_text" for s in qualifier) else 8
        if "quantifier_change" in kinds:
            raw_change_points += 16
        if "negation_change" in kinds:
            raw_change_points += 16
        if any(s["kind"] == "asymptotic_bound_change" and s["action"] == "weaker" for s in strength):
            raw_change_points += 16
        elif any(s["kind"] == "asymptotic_bound_change" and s["action"] == "changed_tight_bound" for s in strength):
            raw_change_points += 8  # A tightened bound alone strengthens the result; no extra points.
        if "inequality_change" in kinds:
            raw_change_points += 14
        elif "numeric_formula_change" in kinds:
            raw_change_points += 12
        elif "formula_change" in kinds:
            raw_change_points += 6
        if kinds & {"formula_added", "formula_omitted"}:
            raw_change_points += 10 if any(s.get("parameter_domain_candidate") for s in strength) else 8
        if "branch_scope_change" in kinds:
            raw_change_points += 8
        if "terminology_change" in kinds:
            raw_change_points += 4
        if "dependency_phrase_change" in kinds:
            raw_change_points += 4
        add("observable_text_changes", min(28, raw_change_points), "按活跃正文可见变化加分，总计封顶28；复合术语与依赖措辞只有低权重，变化多本身不证明同一命题。")
        chronology = algo["chronology"]
        add("chronology", 7 if chronology["statement_order_confirmed"] else 2 if chronology["status"] == "paper_date_only" else -7,
            "实际语句顺序可确定。" if chronology["statement_order_confirmed"] else "仅论文级先后，语句版本顺序未确认。" if chronology["status"] == "paper_date_only" else "先后待定或冲突，不按输入顺序假定修订。")
        disclosure = algo["disclosure"]
        if disclosure["status"] == "matched_correction_language":
            add("correction_language", -16, "窗口含明确纠错措辞，降低未说明变化的调查排序；保留作对照，关联仍待确认。")
        elif disclosure["status"] == "matched_change_language":
            add("change_language", -7, "窗口含一般修订措辞，降低未说明变化的排序；不自动认定已有充分说明。")
        if any(s["kind"] == "restricted_use_context" for s in signals):
            add("restricted_use_retained", 0, "受限使用作为传播线索保留，没有因为是应用或特例而剔除。")
        if serious:
            add("source_ownership_uncertain", -15, "来源归属、多个文档根或停用区域未确定，降低自动排序可靠度；不判定整段为假。")
        support = record["support"]
        if support["distinct_later_papers"] > 1:
            add("repeated_later_paper_candidates", min(12, (support["distinct_later_papers"] - 1) * 3),
                "同一旧文本、同类变化在本批 %d 篇不同后文候选中出现；已按论文去重，尚未证明它们确实沿用该命题。" % support["distinct_later_papers"])
        if support["nonoverlapping_author_groups"] > 1:
            add("different_name_groups", min(4, (support["nonoverlapping_author_groups"] - 1) * 2),
                "本批存在多个与旧作者姓名不重合的组；共著重合组已合并，仍不认证人物独立性。")
        score = sum(c["points"] for c in components)
        if score < 0:
            add("nonnegative_floor", -score, "优先分最低为0；这是未认证的调查队列，不是负概率。")
            score = 0
        if score > 100:
            add("score_ceiling", 100 - score, "优先分上限100。")
            score = 100
    # Alignment gates apply AFTER scoring, with visible deductions. Different
    # objects/headings or closer competing statements cannot win by accumulating
    # more changed words.
    status = alignment["status"]
    cap, reason = 100, ""
    if status in {"named_statement_mismatch", "comparison_object_mismatch", "weak_textual_overlap"}:
        cap, reason = 24, "结果名称不同或共同锚点不足，最高列低优先级；先对齐命题，变化多不能补足对应关系。"
    elif status in {"not_best_statement_candidate", "conclusion_alignment_uncertain", "partial_textual_overlap"}:
        cap, reason = 39, "存在更近竞争声明、结论错配或只有部分对齐，最高列低优先级。"
    elif status == "competing_candidates_ambiguous":
        cap, reason = 49, "同一文档有相近竞争声明，匹配不唯一时最高列中优先级。"
    if serious and cap > 49:
        cap, reason = 49, "来源归属不确定，最高列中优先级。"
    if score > cap:
        add("statement_alignment_gate", cap - score, reason)
        score = cap
    if algo.get("rewrite_equivalence", {}).get("suppresses_strength_alert") and score:
        add("recognized_rewrite_only_gate", -score, "限定规则确认当前仅为等价列举换序/符号文字换写，关闭强度提醒；数学等价规则之外仍未知。")
        score = 0
    algo["score_components"] = components
    algo["priority_score"] = score
    algo["priority_band"] = "high" if score >= 65 else "medium" if score >= 40 else "low" if score > 0 else "no_signal"
    algo["score_interpretation"] = "0—100调查优先级启发式；不是错误概率、弱化概率或经过校准的准确率。"
    if missing:
        headline = "声明缺失，尚不能比较强度变化"
    elif algo.get("rewrite_equivalence", {}).get("suppresses_strength_alert"):
        headline = "限定规则内仅见纯改写；不发强度变化提醒"
    elif algo["text_difference"]["status"] == "formatting_only":
        headline = "当前仅检出排版、注释或标点差异；数学强度仍未知"
    elif algo["text_difference"]["status"] == "notation_only":
        headline = "只检出字体或表示法包装改写；数学作用仍未知"
    elif not strength:
        headline = "当前检测未见有效正文变化；未证明数学强度不变"
    elif status in {"named_statement_mismatch", "comparison_object_mismatch", "weak_textual_overlap", "not_best_statement_candidate", "conclusion_alignment_uncertain", "competing_candidates_ambiguous", "partial_textual_overlap"}:
        headline = alignment["label"]
    elif {s["kind"] for s in strength} <= {"unclassified_text_change"}:
        headline = "活跃正文存在未分类变化；强度与作用域待核"
    elif algo["disclosure"]["status"] == "matched_correction_language":
        headline = "窗口含纠错措辞；保留变化线索与公开更正对照"
    elif any(s["kind"] == "restricted_use_context" for s in signals):
        headline = "出现受限使用与文字变化；尚未认证原命题弱化"
    elif any(s["kind"] == "asymptotic_bound_change" and s["action"] == "weaker" for s in strength):
        headline = "结论中的渐近界放宽（局部比较）；整体数学方向与说明覆盖待核"
    elif any(s["kind"] == "qualifier_change" for s in strength):
        headline = "检出限定条件的文字变化，需核对同一命题的适用范围"
    elif algo.get("new_premises"):
        headline = "同一对象检出新增前提词面；整体数学方向与说明覆盖待核"
    elif any(s["kind"] in {"quantifier_change", "negation_change"} for s in strength):
        headline = "检出量词或否定变化，逻辑强度尚未判定"
    elif all(s["kind"] in {"terminology_change", "dependency_phrase_change"} for s in strength):
        headline = "复合术语或依赖措辞变化，未解析数学强度"
    else:
        headline = "检出公式片段变化，数值保证与逻辑方向尚未判定"
    algo["headline"] = headline


def _candidate_gate(record: dict, serious: bool, missing: bool) -> None:
    a = record["algorithm"]
    alignment = a["statement_alignment"]
    if missing or serious:
        state, label = "source_unverified", "有效来源或声明边界待核"
    elif a.get("rewrite_equivalence", {}).get("suppresses_strength_alert"):
        state, label = "recognized_rewrite_only", "限定规则确认纯改写，不发强度变化提示"
    elif a["text_difference"]["status"] == "formatting_only":
        state, label = "formatting_observation", "格式/注释观察，不发强度变化提示"
    elif a["text_difference"]["status"] == "notation_only":
        state, label = "notation_observation", "字体/表示法观察，未认定数学等价或强度改变"
    elif a["text_difference"]["status"] == "identical":
        state, label = "no_detected_active_change", "本窗口未检出活跃文字变化，不等于数学强度不变"
    elif alignment not in {"strong_textual_overlap", "same_extracted_text", "branch_expansion_textual_match"}:
        state, label = "alignment_pending", "命题对齐待定，暂不进入强度提示队列"
    elif a["chronology"]["direction"] == "unknown":
        state, label = "chronology_pending", "语句方向待定，暂不按修订解释"
    elif any(s["kind"] == "unclassified_text_change" for s in a["signals"]):
        state, label = "unclassified_active_change", "正文变化尚未归类，保留补写/作用域核验队列"
    elif any(v["role"] in {"external_attribution_cue", "conjecture_cue", "definition_cue"} or v.get("opening_attribution_cue") for v in a["assertion_role"].values()):
        state, label = "attribution_pending", "可核文字变化；原创归属或声明角色待核"
    else:
        state, label = "strength_text_candidate", "通过文字对齐门控的变化候选；逻辑强度仍未知"
    eligible = state in {"strength_text_candidate", "attribution_pending"}
    a.update(candidate_state=state, candidate_state_label=label,
             eligible_for_strength_queue=eligible,
             eligible_for_self_revision_queue=bool(eligible and
                 a["author_relation"]["kind"] in {"same_paper", "name_overlap"}),
             queue_gate_meaning="仅确定下一阶段核验队列，不认证同一命题、作者归属、历史变化或逻辑弱化。")
    passed, pending = [], ["同一数学命题与蕴含方向未认证", "命题原创归属未认证", "人物消歧与逐版本作者名单未核"]
    if a["author_relation"]["kind"] == "same_paper":
        passed.append("输入元数据标为同篇论文")
    elif a["author_relation"]["kind"] == "name_overlap":
        passed.append("论文元数据作者姓名有重合")
    else:
        pending.append("缺少同篇或作者姓名关联")
    if eligible:
        passed.append("有效文字变化通过当前命题对齐门控")
    else:
        pending.append("来源、文字对齐、时间方向或变化类型仍在前置队列")
    if not a["chronology"]["statement_order_confirmed"]:
        pending.append("目前只有论文级时间或缺失语句快照顺序")
    if any(v["role"] == "citation_in_statement" for v in a["assertion_role"].values()):
        pending.append("引用可见位置已列出，指向范围与归属仍未知")
    a["self_revision_basis"] = {
        "status": "investigation_route_only", "passed": passed, "pending": pending,
        "meaning": "自身修订队列只表示同篇或作者关联的文本变化调查路由；不是作者原创命题、参与两端版本或修改错误定理的认证。",
        "assertion_original_ownership": "unknown", "author_participation_both_versions": "unknown",
    }


def _next_step(record: dict, serious: bool, missing: bool) -> dict:
    a = record["algorithm"]
    alignment = a["statement_alignment"]
    if missing or serious:
        action, label, reason = "verify_source", "先核对有效原文与声明边界", "当前来源、有效文档归属或声明完整性不足，不能先解释强度。"
    elif a.get("rewrite_equivalence", {}).get("suppresses_strength_alert"):
        action, label, reason = "retain_rewrite_observation", "保留纯改写观察", "限定等价列举/符号字典规则已覆盖全部有效变化；不据此宣称其他语义等价。"
    elif alignment in {"named_statement_mismatch", "comparison_object_mismatch", "not_best_statement_candidate", "competing_candidates_ambiguous", "conclusion_alignment_uncertain", "weak_textual_overlap", "partial_textual_overlap"}:
        action, label, reason = "align_statements", "先对齐同一数学命题", "先比较结果名称、共同对象、结论与竞争候选；条件词差异无法替代命题对应。"
    elif not a["chronology"]["statement_order_confirmed"]:
        action, label, reason = "verify_statement_versions", "核对两段语句的实际版本与日期", "论文首次上传顺序不足以确定未知源码快照中的语句先后。"
    elif any(v["role"] in {"citation_in_statement", "external_attribution_cue", "conjecture_cue"} for v in a["assertion_role"].values()):
        action, label, reason = "verify_assertion_role", "核对命题的引用来源与声明角色", "论文作者关系不等于命题归属；需区分引用、猜想与作者自身主张。"
    elif a["text_difference"]["status"] != "active_text_changed":
        action, label, reason = "no_strength_change_detected_in_window", "本窗口暂未检出强度变化线索", "只覆盖有界声明；排版一致或没有命名信号不能证明全局假设与强度不变。"
    elif any(s["kind"] == "unclassified_text_change" for s in a["signals"]):
        action, label, reason = "inspect_unclassified_change", "检查未分类补写与全局参数范围", "正文存在有效变化，尚未被规则归类；先确认它是否已在全局约定中给出。"
    else:
        action, label, reason = "check_strength_and_disclosure", "核对前提、结论和全局定义，再查说明覆盖", "当前只证实文字差异，数学蕴含和是否已有充分说明均未知。"
    return {"action": action, "label": label, "reason": reason, "automatic": True,
            "selection_basis": "deterministic_evidence_gates", "executed": False,
            "meaning": "算法自动选择的下一步核验任务；不是模型/人工回写的算法标签。"}


def _aggregate(records: list[dict], structures: list[list[str]]) -> None:
    groups = defaultdict(list)
    for i, (record, structure) in enumerate(zip(records, structures)):
        old = record["old"]
        old_hash = sha256(_normalized(old["text"]).encode("utf-8")).hexdigest()[:24]
        key = sha256((_paper_id(old) + "\0" + old_hash + "\0" + "|".join(sorted(structure))).encode("utf-8")).hexdigest()[:24]
        groups[key].append(i)
    for group_key, indices in groups.items():
        eligible = []
        papers = defaultdict(set)
        candidates_by_paper = defaultdict(int)
        old_names = set()
        same_author_papers = set()
        third_party_papers = set()
        version_observations = set()
        raw_later_papers, raw_candidate_papers = set(), set()
        for i in indices:
            record, algo = records[i], records[i]["algorithm"]
            old_names.update(_names(record["old"]))
            chronology = algo["chronology"]
            newpid = _paper_id(record["new"])
            if newpid and newpid != _paper_id(record["old"]):
                raw_candidate_papers.add(newpid)
                if chronology["direction"] != "unknown" and chronology["status"] != "conflict":
                    raw_later_papers.add(newpid)
            if chronology["status"] == "version_order":
                version_observations.add((newpid, _version(record["new"])))
            strength = any(s["kind"] not in {"restricted_use_context", "citation_use_context", "notation_change"} for s in algo["signals"])
            if (not strength or chronology["direction"] == "unknown"
                    or chronology["status"] == "conflict" or not newpid or newpid == _paper_id(record["old"])
                    or not algo.get("eligible_for_strength_queue")):
                continue
            # All are candidate observations, not certified theorem adoptions.
            eligible.append(i)
            papers[newpid].update(_names(record["new"]))
            candidates_by_paper[newpid] += 1
            relation = algo["author_relation"]["kind"]
            if relation == "name_overlap":
                same_author_papers.add(newpid)
            elif relation == "no_name_overlap":
                third_party_papers.add(newpid)
        parent = {paper: paper for paper in papers}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        name_owner = {}
        for paper, names in papers.items():
            for name in names:
                if name in name_owner:
                    parent[find(paper)] = find(name_owner[name])
                else:
                    name_owner[name] = paper
        author_sets = defaultdict(set)
        author_papers = defaultdict(set)
        for paper, names in papers.items():
            if names:
                author_sets[find(paper)].update(names)
                author_papers[find(paper)].add(paper)
        author_groups = []
        for root, names in author_sets.items():
            author_groups.append({
                "group_id": sha256("|".join(sorted(names)).encode("utf-8")).hexdigest()[:20],
                "paper_ids": sorted(author_papers[root]), "normalized_names": sorted(names),
                "overlaps_old_names": bool(names & old_names),
                "overlap_names": sorted(names & old_names),
                "identity_or_independence_certified": False,
            })
        support = {
            "group_key": group_key, "candidate_count": len(indices),
            "distinct_later_papers": len(papers), "later_paper_ids": sorted(papers),
            "raw_observed_distinct_later_papers": len(raw_later_papers),
            "raw_observed_later_paper_ids": sorted(raw_later_papers),
            "raw_observed_distinct_candidate_papers": len(raw_candidate_papers),
            "raw_observed_candidate_paper_ids": sorted(raw_candidate_papers),
            "qualified_distinct_later_papers": len(papers), "qualified_later_paper_ids": sorted(papers),
            "verified_adoption": None, "verified_adoption_status": "not_verified",
            "count_alias_note": "distinct_later_papers兼容字段等于qualified_distinct_later_papers；0合格不表示0原始观察。",
            "qualification_criteria": "eligible_for_strength_queue=true且有可用后文方向、不同论文ID；仍只是候选文字观察，不是已核沿用。",
            "distinct_author_groups": len(author_sets),
            "nonoverlapping_author_groups": sum(not bool(names & old_names) for names in author_sets.values()),
            "same_author_name_later_papers": len(same_author_papers),
            "third_party_name_later_papers": len(third_party_papers),
            "same_author_name_later_paper_ids": sorted(same_author_papers),
            "third_party_name_later_paper_ids": sorted(third_party_papers),
            "author_groups": sorted(author_groups, key=lambda x: x["group_id"]),
            "papers_without_author_metadata": sum(not bool(names) for names in papers.values()),
            "distinct_version_observations": len(version_observations),
            "eligible_candidate_count": len(eligible),
            "author_group_basis": "connected_components_of_overlapping_normalized_names_not_verified_people",
            "add_across_rows": False,
            "deduplicate_across_rows_by": "group_key",
            "counting_unit": "unique_candidate_paper_ids_within_one_group_not_certified_adoptions",
            "population_denominator": "unknown",
            "scope": "仅当前输入：同一旧文本与同类文字变化的候选组；通过文字对齐门控后按论文去重，姓名重合共著组已合并。每行重复同组统计，必须按group_key去重，跨行不能相加。论文先后可能只依据首次上传日期；未认证同一命题、沿用关系、独立作者身份或全库广泛使用。",
            "raw_observation_scope": "本批同组不同目标论文原始观察；later只按可用论文/版本方向，不认证语句快照先后。合格数应用文字对齐门控；经核沿用数未知(null)。",
            "corpus_prevalence_estimated": False,
        }
        for i in indices:
            records[i]["support"] = dict(support)


def _rewrite_canonical(text: str, universal_prefix: bool = False) -> tuple[str, bool]:
    value = _active_text(text)
    value = re.sub(r"\\label\s*\{[^{}]*\}", " ", value)
    value = _ENV.sub(" ", _STYLE.sub(" ", value))
    value = re.sub(r"\\(mathbb|mathcal|mathbf|mathsf|mathfrak)\s+([A-Za-z0-9])", r"\\\1{\2}", value)
    value = re.sub(r"([_^])\s*\{([A-Za-z0-9])\}", r"\1\2", value)
    original = value
    value = re.sub(r"\\(?:text|textrm)\s*\{\s*(?:is\s+|are\s+)?isomorphic\s+to\s*\}", " zzisomorphictozz ", value, flags=re.I)
    value = re.sub(r"\b(?:(?:is|are)\s+)?isomorphic\s+to\b|\\cong(?![A-Za-z])|≅", " zzisomorphictozz ", value, flags=re.I)
    symbol_rule = value != original
    value = _EQUIV_INTRO.sub(" zzequivalentconditionszz ", value)
    if universal_prefix and re.search(r"\bfor\s+(?:any|every|all|each)\b", value, re.I):
        # A repeated universal point binder in an introductory declaration is
        # interchangeable with its comma-separated membership notation. This
        # never maps For any to Let or removes an existential/some binder.
        value = re.sub(r"\band\s+any\s+point\s+(?=\$[^$]{1,100}\\in\b)", " ", value, flags=re.I)
    value = re.sub(r"\\\\(?:\s*\[[^\]]*\])?", " ", value)
    # Preserve mathematical case: A and a are not interchangeable parameters.
    # We may fold ordinary prose, but not commands, math blocks or one-letter
    # tokens (which can also be variables outside math delimiters).
    math_spans = [(m.start(), m.end()) for m in _MATH_BLOCK.finditer(value)]
    tokens = []
    span_index = 0
    for match in _TOKEN.finditer(value):
        while span_index < len(math_spans) and math_spans[span_index][1] <= match.start():
            span_index += 1
        in_math = span_index < len(math_spans) and math_spans[span_index][0] <= match.start() < math_spans[span_index][1]
        token = match.group(0)
        tokens.append(token if in_math or token.startswith("\\") or (len(token) == 1 and token.isalpha()) else token.casefold())
    return " ".join(tokens), symbol_rule


def _equivalent_list(text: str) -> dict | None:
    active = _active_text(text)
    intro = _EQUIV_INTRO.search(active)
    if not intro:
        return None
    begin = re.search(r"\\begin\s*\{(itemize|enumerate)\}", active[intro.end():], re.I)
    if begin:
        start = intro.end() + begin.start()
        body_start = intro.end() + begin.end()
        close = re.search(r"\\end\s*\{" + begin.group(1) + r"\}", active[body_start:], re.I)
        if not close:
            return None
        body_end = body_start + close.start()
        end = body_start + close.end()
        body = active[body_start:body_end]
        if re.search(r"\\begin\s*\{(?:itemize|enumerate)\}", body, re.I):
            return None  # Nested lists are not silently flattened or sorted.
        markers = list(re.finditer(r"\\item(?:\s*\[([^\]]*)\])?", body))
        if any(m.group(1) and not re.fullmatch(r"\s*\(?\s*(?:[ivxlcdm]+|\d+)\s*\)?[.)]?\s*", m.group(1), re.I) for m in markers):
            return None  # A custom item label may carry mathematical data.
        clauses = [body[m.end():markers[i + 1].start() if i + 1 < len(markers) else len(body)] for i, m in enumerate(markers)]
        suffix = active[end:]
    else:
        start = intro.end()
        body = active[start:]
        markers = list(_INLINE_ITEM.finditer(body))
        if len(markers) < 2 or len(markers) > 20:
            return None
        expected = [str(i + 1) for i in range(len(markers))]
        roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"][:len(markers)]
        labels = [m.group(1).casefold() for m in markers]
        if labels != expected and labels != roman:
            return None
        clauses = [body[m.end():markers[i + 1].start() if i + 1 < len(markers) else len(body)] for i, m in enumerate(markers)]
        suffix = ""
    if not 2 <= len(clauses) <= 20:
        return None
    if any(re.search(r"\b(?:preceding|previous|next|former|latter)\b|\b(?:condition|statement|assertion)\s*\(?\s*(?:[ivxlcdm]+|\d+)\s*\)?", c, re.I) for c in clauses):
        return None  # Local ordinal references make order semantically active.
    canonical, used_symbol_rules = [], False
    for clause in clauses:
        clause = re.sub(r"(?:[;,.\s]|\\\\)+$", "", clause)
        clause = re.sub(r"[;,\s]+and\s*$", "", clause, flags=re.I)
        value, used = _rewrite_canonical(clause)
        if not value:
            return None
        canonical.append(value)
        used_symbol_rules |= used
    prefix, prefix_rule = _rewrite_canonical(active[:start], universal_prefix=True)
    suffix_key, suffix_rule = _rewrite_canonical(suffix)
    return {"prefix": prefix, "suffix": suffix_key, "clauses": canonical,
            "symbol_rule_used": used_symbol_rules or prefix_rule or suffix_rule}


def _rewrite_equivalence(old: str, new: str) -> dict:
    a, au = _rewrite_canonical(old)
    b, bu = _rewrite_canonical(new)
    kinds, rule_details = [], []
    if a == b and old != new and (au or bu):
        kinds.append("finite_symbol_word_dictionary")
        rule_details.append("只使用明列的cong/≅/isomorphic to及少量确定TeX写法；否定、参数与不等号保留。")
    al, bl = _equivalent_list(old), _equivalent_list(new)
    if (al and bl and al["prefix"] == bl["prefix"] and al["suffix"] == bl["suffix"]
            and Counter(al["clauses"]) == Counter(bl["clauses"])):
        if al["clauses"] != bl["clauses"]:
            kinds.append("explicit_equivalent_conditions_reordered")
        elif old != new:
            kinds.append("explicit_equivalent_conditions_reformatted")
        if al["symbol_rule_used"] or bl["symbol_rule_used"]:
            kinds.append("finite_symbol_word_dictionary")
        rule_details.append("只对明说互相等价的扁平条件列举比较多重集；重复条款计数保留，不把任意列表当无序。")
    recognized = bool(kinds)
    return {"status": "recognized_rewrite_only" if recognized else "not_established",
            "kinds": sorted(set(kinds)), "suppresses_strength_alert": recognized,
            "label": "限定规则内仅为等价列举换序/符号文字换写，不发强度提醒" if recognized else "未由限定改写规则确认仅是等价换写",
            "rule_details": rule_details, "repeated_clauses_preserved": True,
            "mathematical_equivalence_beyond_supported_rules": "unknown"}


def _investigation_route(record: dict, serious: bool, missing: bool) -> None:
    a = record["algorithm"]
    associated = a["author_relation"]["kind"] in {"same_paper", "name_overlap"}
    premises = a["new_premises"]
    reasons = []
    if missing or serious:
        status, rank, label = "source_unverified", 8, "来源或声明边界待核"
        reasons.append({"key": "source_gate", "reason": "有具体来源/有效文本风险，先核对原文。"})
    elif a["rewrite_equivalence"]["suppresses_strength_alert"]:
        status, rank, label = "rewrite_only", 9, "限定规则内的纯改写；不进强度提醒"
        reasons.append({"key": "rewrite_filter", "reason": a["rewrite_equivalence"]["label"]})
    elif a["text_difference"]["status"] in {"formatting_only", "identical", "notation_only"}:
        status, rank, label = "no_active_change", 10, "未检出有效强度文字线索或只见表示法观察"
        reasons.append({"key": "no_active_change", "reason": "当前范围的排版/记号/一致观察不能证明数学强度不变。"})
    elif not a["eligible_for_strength_queue"]:
        status, rank, label = "alignment_pending", 7, "命题对齐、时间方向或变化归类仍待定"
        reasons.append({"key": "alignment_gate", "reason": a["candidate_state_label"]})
    elif not associated:
        status, rank, label = "third_party_auxiliary", 5, "第三方文字/受限使用辅助线索"
        reasons.append({"key": "third_party_retained", "reason": "不同作者的重复表述仍保留并按论文去重聚合；未认证采用关系。"})
    elif not premises:
        category = a['change_analysis']['category']
        status = category
        rank = 3 if category in {'formula_scope_change','condition_reference_change','quantifier_scope_change'} else 6
        label = a['change_analysis']['label']
        reasons.append({"key": "concrete_change_type", "reason": label})
    else:
        disclosure = a["disclosure_evidence"]
        ds = disclosure["status"]
        reasons.extend([{"key": "author_link", "reason": "同篇或论文元数据姓名关联；人物与原创归属仍未认证。"},
            {"key": "aligned_added_premise", "reason": "先匹配区分大小写的同一对象与句法位置，再检出新增属性/范围文字。"}])
        if ds == "not_found_in_checked_material":
            status, rank, label = "primary_unexplained_tightening", 1, "优先调查：同作者关联新增前提，已查范围未见说明"
            if any(s.get('extent')=='local_following_formula' for p in premises for s in p.get('new_guarantee_scopes',[])):
                label='优先调查：作者关联的局部估计/保证新增条件，已查范围未见说明'
        elif ds == "paper_correction_scope_unconfirmed":
            status, rank, label = "paper_correction_unconfirmed", 2, "新增前提候选；论文更正是否覆盖本条待核"
        elif ds == "material_missing":
            status, rank, label = "disclosure_material_missing", 3, "新增前提候选；说明材料缺失，暂不判未说明"
        elif disclosure["explains_added_premise"]:
            status, rank, label = "explained_change_control", 4, "本条已有修订提示；保留作说明覆盖对照"
        else:
            status, rank, label = "statement_notice_strength_unconfirmed", 2, "本条有修订提示，是否说明新增前提待核"
        reasons.append({"key": "disclosure_coverage", "reason": disclosure["label"]})
        if ds == "statement_explanation_found" and not disclosure["explains_added_premise"]:
            reasons.append({"key": "notice_not_strength_explanation", "reason": "关联说明可能只谈证明或一般调整；尚未确认它说明新增前提或强度变化。"})
        if a["tightening_textual_evidence"]["conclusion_match"]["status"] == "mixed_or_unverified":
            reasons.append({"key": "mixed_change", "reason": "条件/结论还有混合变化，整体逻辑方向未定。"})
        if any(s.get('extent')=='local_following_formula' for p in premises for s in p.get('new_guarantee_scopes',[])):
            reasons.append({'key':'local_guarantee_scope','reason':'新增条件关联同分项中后续的某条公式；不能把较早的求解范围当作它的旧条件，也不外推为整条命题收缩。'})
    primary = status == "primary_unexplained_tightening"
    a["investigation_queue"] = {"status": status, "label": label, "primary": primary, "rank": rank,
        "rank_order": "ascending_rank_then_legacy_priority_score_descending",
        "meaning": "确定性调查路由；不认证数学弱化、错误、原创归属或全渠道悄然修改。"}
    a["eligible_for_primary_investigation_queue"] = primary
    scope_categories={'premise_change','formula_scope_change','condition_reference_change','quantifier_scope_change'}
    a['scope_review_required']=bool(not a['rewrite_equivalence']['suppresses_strength_alert'] and a['change_analysis']['category'] in scope_categories)
    a['scope_evidence_behind_gate']=bool(a['scope_review_required'] and not a['eligible_for_strength_queue'])
    a['scope_review_gate_note']='检测到范围线索，但旧完整候选框的对齐/来源阻断仍保留；先核对是否同一命题' if a['scope_evidence_behind_gate'] else '没有把范围线索自动当作命题对应认证'
    a["priority_reasons"] = reasons
    a["priority_order_basis"] = "investigation_queue.rank_first_legacy_priority_score_secondary_not_probability"


def analyze_pairs(pairs: Iterable[dict]) -> list[dict]:
    """Return deterministic, explained triage cards for only the supplied pairs.

    Input may be a generator.  Runtime is linear in candidate count apart from a
    small bounded comparison of formulas inside each pair; there is no pairwise
    candidate comparison, source-file access, retrieval, or model involvement.
    """
    records, structures, quality, evidence_cache = [], [], [], {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise TypeError("each pair must be a dict")
        old = pair.get("old") if isinstance(pair.get("old"), dict) else {}
        new = pair.get("new") if isinstance(pair.get("new"), dict) else {}
        chronology, reverse = _chronology(old, new)
        if reverse:
            old, new = new, old
        chronology["paper_first_submission_dates"] = {"old": old.get("first_submission_date"), "new": new.get("first_submission_date")}
        chronology["statement_snapshot_dates"] = {
            "old": old.get("date") if _snapshot_date(old) else None,
            "new": new.get("date") if _snapshot_date(new) else None,
        }
        chronology["statement_snapshot_date_status"] = "both_provided" if all(chronology["statement_snapshot_dates"].values()) else "one_or_both_unknown"
        oldtext, oldslice, oldwarnings, oldserious = _statement(old)
        newtext, newslice, newwarnings, newserious = _statement(new)
        rewrite = _rewrite_equivalence(oldtext, newtext)
        old_context,new_context=old.get('static_tex_context'),new.get('static_tex_context')
        if not rewrite['suppresses_strength_alert']:
            ap,ap_info=condition_rewrite_projection(_active_text(oldtext),_math_units,_formula_evidence,old_context)
            bp,bp_info=condition_rewrite_projection(_active_text(newtext),_math_units,_formula_evidence,new_context)
            if (oldtext!=newtext and _rewrite_canonical(ap)[0]==_rewrite_canonical(bp)[0]
                    and any(ap_info.get(k) or bp_info.get(k) for k in ('flat_intervals_projected','static_macros_used','entailed_type_properties_projected','adverb_spelling_variants'))):
                rewrite.update(status='recognized_rewrite_only',suppresses_strength_alert=True,
                    kinds=['source_backed_literal_or_color_alias_and_flat_interval_rewrite'],
                    label='有限宏/颜色、同一区间、拼写或类型已包含性质的换写覆盖了全部差异',
                    rule_details=['只接受来源中位于声明之前、未重定义且不在条件分支中的明列字面别名/颜色恒等包装；简单数值区间保持端点开闭与对象。','复流形类型已包含光滑性；只对完整差异能被明列规则覆盖时过滤。'],
                    normalization_coverage={'old':ap_info,'new':bp_info})
        formula_scope = analyze_formula_scope(_active_text(oldtext), _active_text(newtext), _math_units, _formula_evidence,old_context,new_context)
        premises = _new_premises(oldtext, newtext) + added_formula_premises(formula_scope, oldtext, newtext, _math_units)
        for premise in premises:
            premise["temporal_direction"] = "unoriented_endpoint_difference" if chronology["direction"] == "unknown" else "provided_order_not_mathematical_entailment"
        tightening = _tightening_evidence(oldtext, newtext, premises, rewrite)
        document_disclosure, evidence_documents = _disclosure_documents(
            pair, old, new, oldtext, newtext, premises, reverse, evidence_cache)
        similarity = _similarity(oldtext, newtext)
        signals = _qualifier_signals(oldtext, newtext) + _polarity_signals(oldtext, newtext)
        signals += _formula_signals(oldtext, newtext, similarity)
        signals += _usage(new, newtext)
        if not rewrite['suppresses_strength_alert']:
            signals += _asymptotic_signals(oldtext, newtext)
        if not rewrite['suppresses_strength_alert']:
            for finding in formula_scope['findings']:
                if finding.get('eligible_as_added_premise'):
                    continue  # The premise signal below already exposes it.
                signals.append(_signal('formula_scope_evidence', finding['label'],
                    finding.get('old_quote',''), finding.get('new_quote',''), finding['comparison_rule'],
                    finding.get('syntactic_role','statement_text'), action=finding['kind'],
                    direction_hint='local_'+finding.get('local_condition_direction','not_established'),
                    old_span=finding.get('old_span'),new_span=finding.get('new_span'),
                    span_unit='characters_within_extracted_statement',
                    logical_role_status='bounded_syntax_not_global_entailment',
                    change_signature=finding.get('finding_id',finding['kind'])))
        for premise in premises:
            signals.append(_signal("aligned_premise_addition", premise["label"],
                premise["old_minimal"], premise["new_minimal"],
                "同一词面对象在前提句法位置新增属性/范围文字；只作收紧候选，未核全局约定、对象数学同一性或蕴含方向。",
                premise["scope"], term=premise["property"], action="added_property_on_aligned_object",
                direction_hint="syntactic_attribute_addition_not_certified_weakening",
                old_span=premise["old_span"], new_span=premise["new_span"], span_unit=premise["span_unit"],
                premise_id=premise["premise_id"], object_key=premise["object_key"], logical_role_status="syntactic_premise_not_semantic_certificate"))
        text_difference = _text_difference(oldtext, newtext)
        change_analysis = classify_change(formula_scope, signals, premises, rewrite, text_difference)
        named_signals = [s for s in signals if s["kind"] not in {"restricted_use_context", "citation_use_context", "notation_change"}]
        if text_difference["status"] == "active_text_changed" and not named_signals:
            first_change = text_difference["changes"][0] if text_difference["changes"] else {}
            signals.append(_signal(
                "unclassified_text_change", "有效正文有未分类变化（并非没有变化）",
                first_change.get("old_quote", ""), first_change.get("new_quote", ""),
                "当前命名规则未归类这处有效正文变化。补写参数范围、分项量词或对象可能有作用，也可能是明确已有约定；数学强度未知。",
                "active_statement_text", action="unclassified_active_change", direction_hint="unresolved_strength_direction",
                quote_source="comment_masked_active_statement", logical_role_status="unknown",
                old_span=first_change.get("old_span"), new_span=first_change.get("new_span"),
                span_unit="characters_within_extracted_statement",
                change_signature=sha256((_string(first_change.get("old_changed_text")) + "\0" + _string(first_change.get("new_changed_text"))).encode("utf-8")).hexdigest()[:20],
            ))
        disclosure = _disclosure(old, new, oldtext, newtext)
        roles = {"old": _assertion_role(old, oldtext), "new": _assertion_role(new, newtext)}
        alignment = _alignment(oldtext, newtext, similarity)
        warnings = list(dict.fromkeys(["第一端：" + s for s in oldwarnings] + ["第二端：" + s for s in newwarnings]))
        if chronology["status"] == "paper_date_only":
            warnings.append("两端依论文日期排序；源版本未明时，不能据首次上传先后断言当前两段语句的历史顺序。")
        elif not chronology["statement_order_confirmed"]:
            warnings.append("语句时间方向未确认：第一端/第二端沿用输入位置，不能称为旧主张/新主张。")
        if chronology["direction"] == "reversed":
            warnings.append("已按版本或日期依据交换输入两端；引用原pair时请注意方向。")
        if pair.get("author_match_hint"):
            warnings.append("作者匹配提示只作未核线索，没有用于认证人物身份。")
        if signals and similarity < .30:
            warnings.append("两端文字重合有限；差异可能来自不同命题，而非同一命题修订。")
        if disclosure["status"] == "matched_correction_language":
            warnings.append("纠错词可能涉及窗口内其他结果；尚未确认它解释当前变化。")
        if any(v["role"] in {"citation_in_statement", "external_attribution_cue"} for v in roles.values()):
            warnings.append("声明有外部引用/归属线索；论文作者相同不等于作者在修改自己的原创命题。")
        if chronology["direction"] == "unknown":
            for signal in signals:
                signal["direction_hint_without_time"] = signal.get("direction_hint")
                signal["direction_hint"] = "unoriented_text_difference"
        for signal in signals:
            identity = "\0".join(_string(value) for value in (
                pair.get("pair_id"), signal["kind"], signal.get("term"), signal.get("action"),
                signal.get("old_quote"), signal.get("new_quote"), signal.get("change_signature"),
                signal.get("old_formula_span"), signal.get("new_formula_span"), signal.get("formula_span"), signal.get("changed_span")))
            signal["signal_id"] = "sig_" + sha256(identity.encode("utf-8")).hexdigest()[:24]
            signal["event_id"] = signal["signal_id"]
            signal["evidence_location_basis"] = "extracted_statement_character_spans_when_available_else_quoted_local_window_not_source_coordinates"
        algorithm = {
            "version": VERSION, "signals": signals, "lexical_similarity": similarity,
            "lexical_similarity_definition": "有界声明词袋的多重集Jaccard；保留量词、否定与不等符号。相似度不是数学等价度。",
            "author_relation": _author_relation(old, new, pair.get("author_match_hint")),
            "chronology": chronology, "disclosure": disclosure, "warnings": warnings,
            "weaker_claim_certified": False,
            "certification_flag_meaning": "weaker_claim_certified=false仅表示未认证；不是已证明没有弱化。",
            "decision_status": {"mathematical_weakening": "unknown", "silent_change": "unknown",
                                "same_mathematical_statement": "unknown", "original_claim_ownership": "unknown",
                                "observed_text_change": "observed" if text_difference["status"] in {"active_text_changed", "notation_only"} else "not_detected"},
            "text_difference": text_difference, "assertion_role": roles,
            "new_premises": premises, "rewrite_equivalence": rewrite,
            "formula_scope_analysis": formula_scope, "change_analysis": change_analysis,
            "tightening_textual_evidence": tightening, "disclosure_evidence": document_disclosure,
            "disclosure_precedence": "disclosure_evidence_is_material_scoped_authoritative_legacy_disclosure_is_local_keyword_compatibility_only",
            "statement_alignment": alignment["status"], "alignment_evidence": alignment,
            "mathematical_entailment_checked": False, "model_or_human_labels_used": False,
        }
        record = {
            "pair_id": pair.get("pair_id"),
            "old": _display_side(old, oldtext, oldslice), "new": _display_side(new, newtext, newslice),
            "algorithm": algorithm,
            "channels": pair.get("channels", []), "retrieval_rank": pair.get("retrieval_rank"),
            "evidence_kind": pair.get("evidence_kind", "unknown"),
            "evidence_documents": evidence_documents,
        }
        if isinstance(pair.get("cached_alignment_competition"), dict):
            record["cached_alignment_competition"] = pair["cached_alignment_competition"]
        if isinstance(pair.get("evidence_collection"), dict):
            record["evidence_collection"] = pair["evidence_collection"]
        for key in ("origin_query_id", "origin_candidate_id", "eval_case_id"):
            if key in pair:
                record[key] = pair[key]
        structure = []
        for signal in signals:
            if signal["kind"] not in {"restricted_use_context", "citation_use_context", "notation_change"}:
                structure.append(":".join(_string(signal.get(key)) for key in ("kind", "term", "action", "change_signature")))
        structures.append(structure)
        records.append(record)
        quality.append((oldserious or newserious, not bool(_tokens(oldtext)) or not bool(_tokens(newtext))))
    _add_competition(records)
    for record, (serious, missing) in zip(records, quality):
        _candidate_gate(record, serious, missing)
    _aggregate(records, structures)
    for record, (serious, missing) in zip(records, quality):
        _score(record, serious, missing)
        record["algorithm"]["next_step"] = _next_step(record, serious, missing)
        _investigation_route(record, serious, missing)
    return records
