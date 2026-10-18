"""Readable, source-linked Markdown for automatically discovered change signals.

This module formats engine output; it makes no per-pair decisions.  Unknown
fields remain readable and missing fields are shown as unknown, rather than
being silently interpreted as evidence.
"""
from __future__ import annotations

import html
import math
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from hashlib import sha256
from typing import Any

VERSION = "1.7.0"
__version__ = VERSION
DEFAULT_MAX_DISPLAY = 30

_DATE_KINDS = {
    "first_submission": "论文首次提交；非该陈述版本日期",
    "first_submitted": "论文首次提交；非该陈述版本日期",
    "paper_first_submission": "论文首次提交；非该陈述版本日期",
    "version_date": "所示版本日期",
    "version_submission": "所示版本提交日期",
    "version": "所示版本日期",
    "official_arxiv_version_history_utc": "官方 arXiv 逐版本提交时间（UTC）",
    "unknown": "日期依据未知",
}
_DISCLOSURE = {
    "matched_correction_language": "给定窗口命中明确更正措辞",
    "matched_change_language": "给定窗口命中一般修订措辞",
    "not_matched_in_provided_window": "仅在给定窗口未命中说明",
    "insufficient_window": "没有足够的说明筛查窗口",
}
_BANDS = {
    "high": "高优先",
    "medium": "中优先",
    "low": "低优先",
    "urgent": "高优先",
    "review": "待核",
    "unranked": "未分级",
    "unknown": "未分级",
    "no_signal": "未命中强度规则",
}

_ALIGNMENT = {
    "same_extracted_text": "规范化后的声明文字一致；未认证数学等价",
    "strong_textual_overlap": "声明文字重合较高；同一数学命题仍未确认",
    "partial_textual_overlap": "声明文字部分重合；同一数学命题仍未确认",
    "weak_textual_overlap": "声明文字对应较弱，先核对是否同一命题",
    "named_statement_mismatch": "两端声明名称不同；同一命题对应待核",
    "competing_candidates_ambiguous": "存在多个竞争候选，命题对应尚未确定",
    "branch_expansion_textual_match": "存在分支展开的文字对应；数学强弱未确认",
    "comparison_object_mismatch": "比较对象尚未对齐，先核对是否同一命题",
}
_USAGE_SIGNALS = {"restricted_use_context", "citation_use_context"}
_FORMAT_SIGNALS = {"formatting_change", "punctuation_change", "punctuation_only_change", "layout_change", "metadata_change"}
_CANDIDATE_STATES = {
    "source_unverified": "先补来源，暂不进入强度候选队列",
    "formatting_observation": "格式变化观察",
    "no_detected_active_change": "未检出声明活动文本变化",
    "alignment_pending": "命题对应待核",
    "condition_clause_correspondence_pending": "条件从句或触发分支对应待核",
    "unclassified_active_change": "活动文本有变化，当前强度规则未归类",
    "attribution_pending": "声明归属待核",
    "strength_text_candidate": "可进入文字强度候选核验队列",
    "chronology_pending": "先后顺序待核",
    "notation_observation": "记号／字体包装变化观察，数学作用未知",
}
_DISCLOSURE_EVIDENCE = {
    "statement_explanation_found": "本条有说明／修订提示（解释程度另列）",
    "paper_correction_scope_unconfirmed": "论文有更正材料，与本条变化的关联未定",
    "not_found_in_checked_material": "已扫描材料范围内未见本条说明",
    "material_missing": "说明资料缺失，不能作未见说明判断",
}
_INVESTIGATION_QUEUES = {
    "primary_unexplained_tightening": "已查范围未见说明的前提文字收紧候选",
    "statement_notice_strength_unconfirmed": "本条有修订提示，是否说明新增前提待核",
    "paper_correction_unconfirmed": "论文有更正、与本条关联未定",
    "disclosure_material_missing": "说明材料待补",
    "explained_change_control": "已说明变化对照",
    "third_party_auxiliary": "第三方辅助线索",
    "other_text_change": "其他文字变化、强度未定",
    "alignment_pending": "命题对应待核",
    "source_unverified": "来源待核",
    "rewrite_only": "纯改写已按规则排除",
    "no_active_change": "未检出活动文本变化",
}
_DOCUMENT_TYPES = {"version_comment": "公开版本说明", "public_erratum": "公开勘误／纠错材料", "full_tex": "完整 TeX 扫描材料", "abstract": "公开摘要"}


def _map(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _plain(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "；".join(_plain(v) for v in value if v is not None)
    if isinstance(value, Mapping):
        return "；".join(f"{k}: {_plain(v)}" for k, v in value.items())
    return str(value)


def _short(value: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", _plain(value)).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _md(value: Any) -> str:
    """Escape input text, including raw HTML, in ordinary Markdown positions."""
    text = html.escape(_plain(value), quote=False)
    text = text.replace("\\", "\\\\")
    for token in ("`", "*", "_", "[", "]", "|", "#", "~", "$"):
        text = text.replace(token, "\\" + token)
    return text.replace("\r", " ").replace("\n", " ")


def _code(value: Any) -> str:
    text = _plain(value).replace("\x00", "�")
    # A source fragment cannot close its own Markdown fence.
    longest = max((len(v) for v in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _score(item: Mapping) -> float:
    value = _map(item.get("algorithm")).get("priority_score")
    try:
        score = float(value)
        return score if math.isfinite(score) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _score_label(item: Mapping) -> str:
    raw = _map(item.get("algorithm")).get("priority_score")
    if raw is None:
        return "未知"
    try:
        score = float(raw)
        return f"{score:g}" if math.isfinite(score) else "未知"
    except (TypeError, ValueError):
        return "未知"


def _core_rank(item: Mapping) -> float:
    value = _map(_map(item.get("algorithm")).get("investigation_queue")).get("rank")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return float("inf")


def _status_label(value: Any, fallback: str = "未知") -> str:
    if isinstance(value, Mapping):
        return _plain(value.get("label") or value.get("status") or fallback)
    return _plain(value) or fallback


def _core_disclosure_label(algorithm: Mapping) -> str:
    evidence = _map(algorithm.get("disclosure_evidence"))
    status = _plain(evidence.get("status"))
    if status == "statement_explanation_found":
        if evidence.get("explains_added_premise") is True:
            return "本条新增前提有明确说明（程序关联）"
        if evidence.get("explains_added_premise") is False:
            return "本条有修订提示，是否说明新增前提待核"
    return _DISCLOSURE_EVIDENCE.get(status) or _plain(evidence.get("label")) or status or "说明资料与关联状态未知"


def summary_counts(results: list) -> dict:
    """Count supplied core categories, without changing eligibility or rank."""
    queues, explanation, change = Counter(), Counter(), Counter()
    for result in results:
        algorithm = _map(_map(result).get("algorithm"))
        queues[_plain(_map(algorithm.get("investigation_queue")).get("status")) or "unknown"] += 1
        explanation[_plain(_map(algorithm.get("disclosure_evidence")).get("status")) or "unknown"] += 1
        change[_plain(_map(algorithm.get("tightening_textual_evidence")).get("status")) or "unknown"] += 1
    changes = Counter(_plain(_map(_map(r.get('algorithm')).get('change_analysis')).get('category')) or 'unavailable' for r in results)
    return {"investigation_queues": dict(queues), "disclosure_states": dict(explanation), "tightening_textual_states": dict(change), 'concrete_change_categories':dict(changes)}


def _items(value: Any) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _paper(paper: Mapping) -> str:
    paper_id = _plain(paper.get("paper_id")) or "论文标识未知"
    version = _plain(paper.get("version")).strip()
    if version:
        version = version if version.lower().startswith("v") else "v" + version
        # Avoid rendering 'idv2 v2' when the identifier already has a version.
        if not paper_id.lower().endswith(version.lower()):
            paper_id += " " + version
    return _md(paper_id)


def _date(paper: Mapping) -> str:
    date = _plain(paper.get("date")).strip()
    if not date:
        first = _plain(paper.get("first_submission_date"))
        label = "版本日期未知" if paper.get("version") else "陈述快照日期未知"
        return label + (f"；论文首发 {_md(first)}" if first else "；论文首发日期也未知")
    kind = _plain(paper.get("date_kind"))
    basis = _DATE_KINDS.get(kind, kind or "日期依据未知")
    return f"{_md(date)}（{_md(basis)}）"


def _version_date(paper: Mapping) -> str:
    if paper.get("version_date"):
        return _md(paper["version_date"]) + "（" + _md(paper.get("version_date_basis") or "所给版本日期") + "）"
    kind = _plain(paper.get("date_kind"))
    if paper.get("date") and kind in {"version_date", "version_submission", "version", "version_submission_date", "exact_version_date", "official_arxiv_version_history_utc"}:
        return _md(paper["date"]) + "（" + _md(_DATE_KINDS.get(kind, kind)) + "）"
    return "版本日期未知；不以论文首发日期替代"


def _authors(paper: Mapping) -> str:
    names = paper.get("authors")
    return _md(_short(names, 280)) if names else "未知"


def _file_link(path: Any, label: str, line: Any = None) -> str:
    path = _plain(path)
    normalized = path.replace("\\", "/")
    is_absolute = normalized.startswith("/") or bool(re.match(r"^[A-Za-z]:/", normalized))
    if not path:
        return _md(label) + "（路径缺失）"
    if not is_absolute:
        return _md(label) + "（未提供绝对路径：" + _md(_short(path, 220)) + "）"
    target = normalized.replace("<", "%3C").replace(">", "%3E")
    target = target.replace("\r", "%0D").replace("\n", "%0A")
    if isinstance(line, int) and not isinstance(line, bool) and line > 0:
        target += ":" + str(line)
    return f"[{_md(label)}](<{target}>)"


def _location(source: Any) -> str:
    source = _map(source)
    path = _plain(source.get("path"))
    span = source.get("span")
    if source.get('viewer_url'):
        label = _url_link(source['viewer_url'], '抽取时记录的来源')
    elif not path:
        label = "无源码路径"
    else:
        label = _file_link(path, "抽取时记录的来源", source.get("line_start", source.get("start_line")))
    if span is not None:
        label += "；原输入偏移 " + _md(str(span)) + "（不是行号；未转换为全文坐标，请用上方完整 TeX 行号）"
    return label


def _source_access_summary(source: Any) -> dict:
    access = _map(_map(source).get("source_access"))
    keys = ("full_tex_available", "path", "sha256", "statement_start_line", "statement_end_line",
            "match_status", "source_scope", "project_root", "statement_match_scope", "source_encoding", "bytes",
            "assembly_receipt", "assembly_receipt_sha256", "selected_project_root", "unresolved_include_syntax",
            "static_assembly_status", "document_roots", "backend", "viewer_url", "ref", "paper_id", "version",
            "corpus_locator", "version_binding", "archive_path", "archive_member", "corpus_fallback_reason")
    result = {key: access.get(key) for key in keys}
    roots = _items(access.get("related_roots"))
    result["related_roots"] = [({key: root[key] for key in ("path", "sha256", "source_scope", "statement_start_line", "statement_end_line") if key in root} if isinstance(root, Mapping) else _plain(root)) for root in roots[:8]]
    result.update(related_roots_total=len(roots), related_roots_truncated=len(roots) > 8)
    return result


def _full_source_ref(source: Any) -> str:
    access = _map(_map(source).get("source_access"))
    available, path = access.get("full_tex_available"), access.get("path")
    if available is False:
        return "完整 TeX 文件不可用／未取得；" + _md(_short(access.get("match_status") or "未定位", 100))
    if available is not True:
        return "完整 TeX 可用性未知；当前节选／索引指针不能当作完整源码"
    if not path and not access.get('viewer_url'):
        return "标记为可用但缺少文件路径，暂无法打开完整 TeX"
    start, end = access.get("statement_start_line"), access.get("statement_end_line")
    valid_start = isinstance(start, int) and not isinstance(start, bool) and start > 0
    label = "打开完整 TeX" + (f" 第 {start} 行" if valid_start else "（声明未定位到行）")
    result = _url_link(access['viewer_url'], label) if access.get('viewer_url') else _file_link(path, label, start if valid_start else None)
    if access.get('backend') == 'corpus':
        result += '；原库按需直读'
    if valid_start and isinstance(end, int) and end >= start:
        result += f"；声明位于第 {start}—{end} 行"
    result += "；定位状态 " + _md(access.get("match_status") or "未知")
    return result


def _url_link(url: Any, label: str = "公开证据") -> str:
    value = _plain(url).strip()
    if not re.match(r"^https?://[^\s<>]+$", value, re.I):
        return _md(value) if value else ""
    return f"[{_md(label)}](<{value}>)"


def _documents(item: Mapping) -> list[Mapping]:
    raw = item.get("evidence_documents")
    if raw is None:
        raw = _map(item.get("algorithm")).get("evidence_documents")
    return [_map(doc) for doc in _items(raw) if _map(doc)]


def _compact_document(doc: Mapping, matches: list | None = None) -> dict:
    keys = ("doc_id", "source_type", "paper_id", "version", "source_paper_id", "source_version", "side", "source_path", "url", "sha256",
            "coverage_scope", "source_available", "identity_basis", "association_scope", "statement_reference_keys",
            "source_char_count", "text_truncated", "text_scope", "scanned_characters", "association_status",
            "statement_match_status", "association_reason", "notice_kind", "viewer_url", "source_storage")
    result = {key: doc.get(key) for key in keys if key in doc}
    if result.get('source_path'):
        result['original_source_path']=result['source_path']
        result['source_path']=str(Path(result['source_path']).resolve())
    result['physical_source_group']=_source_group(doc)
    result['source_roles_are_independent_evidence']=False
    text = _plain(doc.get("text"))
    matching_quotes = []
    for raw in matches or []:
        match = _map(raw)
        if doc.get("doc_id") is not None and match.get("doc_id", match.get("document_id")) == doc.get("doc_id"):
            quote = match.get("quote") or match.get("excerpt") or match.get("matched_text") or match.get("text")
            if quote:
                matching_quotes.append(_plain(quote))
    selected = "\n".join(matching_quotes[:2]) if matching_quotes else text if doc.get("source_type") != "full_tex" else ""
    # Document bodies may be megabytes. Only normalize an already bounded slice.
    prefix = re.sub(r"\s+", " ", selected[:1600]).strip()
    excerpt = prefix if len(prefix) <= 400 else prefix[:399] + "…"
    result.update(excerpt=excerpt, input_text_characters=len(text), excerpt_displayed_characters=len(excerpt),
                  excerpt_truncated=len(selected) > 1600 or len(prefix) > 400,
                  excerpt_selection="algorithm_matched_evidence" if matching_quotes else "bounded_document_opening" if selected else "full_tex_body_omitted_use_pointer",
                  excerpt_normalization="whitespace_collapsed", full_document_in_compact=False)
    return result


def _compact_disclosure_evidence(value: Any) -> dict:
    evidence = _map(value)
    result = {key: evidence.get(key) for key in ("status", "label", "coverage", "silent_change", "explains_added_premise", "statement_notice_strength_scopes")}
    matched = [_map(item) for item in _items(evidence.get("matched_evidence")) if _map(item)]
    result["matched_evidence"] = []
    groups={}
    for match in matched:
        source_key=(_source_group(match),_plain(match.get('quote')))
        if source_key in groups:
            groups[source_key]['source_roles'].append(match.get('source_type'))
            groups[source_key]['doc_ids'].append(match.get('doc_id'))
            continue
        row = dict(match)
        row['physical_source_group']=source_key[0]
        row['source_roles']=[match.get('source_type')]
        row['doc_ids']=[match.get('doc_id')]
        row['source_roles_are_independent_evidence']=False
        if row.get('source_path'):
            row['original_source_path']=row['source_path'];row['source_path']=str(Path(row['source_path']).resolve())
        for key in ("quote", "text", "excerpt", "matched_text"):
            if isinstance(row.get(key), str):
                original = row[key]
                row[key] = original[:800]
                row[key + "_displayed_characters"] = len(row[key])
                row.setdefault(key + "_total_characters", len(original))
                total = row.get(key + "_total_characters")
                row[key + "_truncated"] = row.get(key + "_truncated") is True or len(original) > 800 or (isinstance(total, int) and total > len(row[key]))
        groups[source_key]=row
    result['matched_evidence']=list(groups.values())[:8]
    result['distinct_physical_source_groups']=len({k[0] for k in groups})
    result['duplicate_source_role_records_merged']=len(matched)-len(groups)
    result.update(matched_evidence_available=len(matched), matched_evidence_displayed=len(result["matched_evidence"]))
    return result

def _source_group(doc):
    source=doc.get('sha256') or doc.get('doc_id') or doc.get('url') or 'unknown'
    return 'source_'+sha256(str(source).encode()).hexdigest()[:20]


def _evidence_lines(item: Mapping) -> list[str]:
    algorithm = _map(item.get("algorithm"))
    evidence = _map(algorithm.get("disclosure_evidence"))
    matched = _items(evidence.get("matched_evidence"))
    documents = _documents(item)
    lines = []
    matched_seen=set()
    for raw in matched:
        match = _map(raw)
        quote = match.get("quote") or match.get("excerpt") or match.get("matched_text") or match.get("text")
        marker=(_source_group(match),_plain(quote))
        if marker in matched_seen:continue
        matched_seen.add(marker)
        if len(matched_seen)>2:break
        if quote:
            lines.extend(["说明证据摘录（" + _md(match.get("doc_id", match.get("document_id", "文档标识未知"))) + "）：" + _md(_short(_plain(quote)[:2000], 430)), ""])
    if not documents:
        return lines + ["说明材料清单未提供；不能把局部窗口未命中当成已检查全文。", ""]
    physical_groups={_source_group(d) for d in documents}
    lines.extend(["<details>", f"<summary>查看 {len(documents)} 条用途记录、{len(physical_groups)} 组实际来源的入口与短摘录</summary>", "", "同一文件/摘要可同时用于版本说明、摘要和公开纠错线索；这些用途不算多份独立证据。以下完整内容不在精简 JSON 中。", ""])
    for doc in documents[:8]:
        compact = _compact_document(doc, matched)
        source_type = _plain(doc.get("source_type"))
        lines.append("- **" + _md(doc.get("doc_id") or "文档标识未知") + "** · " + _md(_DOCUMENT_TYPES.get(source_type, source_type or "类型未知")) + " · " + _md(doc.get("paper_id") or "论文未知") + " " + _md(doc.get("version") or "") + " · 关联范围 " + _md(doc.get("association_scope") or "未知"))
        lines.append('  实际来源组：'+_md(_source_group(doc))+'；同组用途不独立计数。')
        pointers = []
        if doc.get('viewer_url'):
            pointers.append(_url_link(doc['viewer_url'], '打开原扫描证据'))
        elif doc.get("source_path"):
            pointers.append(_file_link(str(Path(doc["source_path"]).resolve()), "打开本地证据"))
        if doc.get("url"):
            pointers.append(_url_link(doc["url"]))
        if pointers:
            lines.append("  " + "；".join(pointers))
        if doc.get("source_available") is False:
            lines.append("  当前材料状态：文件／内容不可用。")
        lines.append("  扫描／提供范围：" + _md(doc.get("coverage_scope") or doc.get("text_scope") or "未知") + "；源字符数 " + _count(doc.get("source_char_count")) + "；提供文本截断状态 " + _md(doc.get("text_truncated") if "text_truncated" in doc else "未知") + "。")
        if compact.get("excerpt"):
            lines.append("  摘录：" + _md(compact["excerpt"]))
        if doc.get("sha256"):
            lines.append("  文件／材料 SHA-256：" + _md(doc["sha256"]))
        lines.append("")
    if len(documents) > 8:
        lines.extend([f"另 {len(documents) - 8} 份材料指针保留在精简 JSON；详细摘录见完整记录。", ""])
    lines.extend(["</details>", ""])
    return lines


def _premise_lines(algorithm: Mapping) -> list[str]:
    premises = [_map(premise) for premise in _items(algorithm.get("new_premises")) if _map(premise)]
    if not premises:
        return []
    lines = ["**新增前提文字（程序识别候选）：**", "", "| 对象／性质 | 原片段 | 新增条件最小片段 | 对象对应／范围 |", "|---|---|---|---|"]
    grouped={}
    scope_notes=[]
    for premise in premises:
        key=(_plain(premise.get('old_minimal')),_plain(premise.get('new_minimal')),premise.get('clause_id'),premise.get('old_anchor_kind'))
        grouped.setdefault(key,[]).append(premise)
    for group in list(grouped.values())[:4]:
        premise=group[0]
        description = '、'.join(_plain(p.get('object_key') or '对象未定') for p in group) + "／" + _plain(premise.get("label") or premise.get("property") or "性质未定")
        old = _plain(premise.get("old_minimal")) or "未提供对应旧片段"
        if premise.get('old_condition_formula_present') is False:
            old='本分项未定位到对应的旧条件公式；旧符号上下文另见JSON，不能拿它当旧适用域'
        new = _plain(premise.get("new_minimal"))
        new_display = "**" + _md(new) + "**" if new else "未提供新增文字"
        alignment = _status_label(premise.get("object_alignment"))
        scope = _plain(premise.get("scope")) or "范围未定"
        if premise.get('clause_id'):scope+='；源码枚举分项序号'+str(premise['clause_id'])
        lines.append("| " + _md(description) + " | " + _md(old) + " | " + new_display + " | " + _md(alignment + "；" + scope) + " |")
        for target_scope in _items(premise.get('new_guarantee_scopes'))[:1]:
            if target_scope.get('extent')=='local_following_formula':
                target=_map(target_scope.get('target'))
                scope_notes.extend(['','**条件附着的局部保证：** '+_md(target_scope['label'])+'；公式标签 '+_code(target.get('tex_labels'))+'。',
                    '对应公式：'+_code(_short(target.get('formula_quote'),500))+'。',
                    '较早公式独立保留：'+_code(_short(_map(target_scope.get('earlier_formula_in_same_item')).get('formula_quote'),500))+'。不能把这两个层次的范围直接当作前后适用域。',''])
    lines.extend(scope_notes)
    if len(premises) > 4:
        lines.extend(["", f"另 {len(premises) - 4} 项前提候选见精简 JSON。"])
    lines.append("")
    return lines


def _count(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return str(value)
    return "未知"


def _window(text: Any, center: int, limit: int) -> str:
    """A bounded visible-text window; extend cuts to keep compound words whole."""
    text = _plain(text)
    if len(text) <= limit:
        return text
    start = max(0, min(center - limit // 2, len(text) - limit))
    end = min(len(text), start + limit)
    if start:
        boundary = re.search(r"\s", text[max(0, start - 120):start])
        if boundary:
            start = max(0, start - 120) + boundary.end()
    if end < len(text):
        boundary = re.search(r"\s", text[end:min(len(text), end + 120)])
        if boundary:
            end += boundary.start()
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def _paired_quotes(old: Any, new: Any, limit: int = 340) -> tuple[str, str]:
    a, b = _plain(old), _plain(new)
    position = 0
    for left, right in zip(a, b):
        if left != right:
            break
        position += 1
    return _window(a, position, limit), _window(b, position, limit)


def _punctuation_only(old: Any, new: Any) -> bool:
    a, b = _plain(old).strip(), _plain(new).strip()
    if not a or not b or a == b:
        return False
    def normalize(text):
        text = re.sub(r"[.,;:]+(?=\s*(?:\$|\\\)|\\\]|$))", "", text)
        return re.sub(r"\s+", " ", text).strip()
    return normalize(a) == normalize(b)


def _main_signals(algorithm: Mapping) -> list[Mapping]:
    if _map(algorithm.get('rewrite_equivalence')).get('suppresses_strength_alert'):
        return []
    return [s for raw in _items(algorithm.get("signals")) if (s := _map(raw))
            and s.get("kind") not in _USAGE_SIGNALS | _FORMAT_SIGNALS
            and not _punctuation_only(s.get("old_quote"), s.get("new_quote"))]


def _result_name(paper: Mapping) -> str:
    for key in ("statement_title", "statement_name", "declaration_title", "result_name"):
        if paper.get(key):
            return _short(paper[key], 240)
    text = _plain(paper.get("text"))[:3000]
    match = re.search(r"\\begin\s*\{(?:theorem|thm|lemma|lem|proposition|prop|corollary|cor|claim|conjecture|conj|definition|defn|remark|rem)\*?\}\s*\[", text, re.I)
    if not match:
        match = re.match(r'\s*\[', text)
    if match:
        depth, start = 1, match.end()
        for end in range(start, min(len(text), start + 700)):
            if text[end] == "[":
                depth += 1
            elif text[end] == "]":
                depth -= 1
                if depth == 0:
                    return _short(text[start:end], 240)
    return ""


def _alignment_label(algorithm: Mapping) -> str:
    evidence = _map(algorithm.get("alignment_evidence"))
    status = _plain(algorithm.get("statement_alignment"))
    label = evidence.get("label") or _ALIGNMENT.get(status) or status or "命题对应资料不足"
    decision = _map(algorithm.get("decision_status")).get("same_mathematical_statement")
    if decision in (None, "unknown") and "未确认" not in _plain(label) and "待核" not in _plain(label):
        label = _plain(label) + "；同一数学命题尚未确认"
    return _plain(label)


def _important_gap(algorithm: Mapping) -> str:
    step = _map(algorithm.get("next_step"))
    if step.get("reason"):
        return _short(step["reason"], 260)
    warnings = [_plain(w) for w in _items(algorithm.get("warnings")) if _plain(w)]
    if warnings:
        return _short(warnings[0], 260)
    return "尚未核实同一命题的全局前提及数学蕴含方向"


def _next_step(algorithm: Mapping) -> str:
    step = algorithm.get("next_step")
    if isinstance(step, Mapping):
        return _plain(step.get("label") or step.get("action") or "先核对命题对应与来源，再比较强度")
    return _plain(step) or "先核对命题对应、声明归属与版本，再比较前提和结论"


def _difference_label(algorithm: Mapping, old: Mapping, new: Mapping) -> str:
    if _map(algorithm.get("rewrite_equivalence")).get("status") == "recognized_rewrite_only":
        return "文字差异已在规则范围内归为纯改写，未形成新增前提候选"
    status = _map(algorithm.get("text_difference")).get("status")
    if status == "identical":
        return "给定声明未见活动文本差异；声明外变化未查"
    if status == "formatting_only":
        return "仅见排版／标点差异，未形成强度线索"
    if status == "notation_only":
        return "仅检出记号／字体包装变化，数学作用未知"
    if _main_signals(algorithm):
        return "检出可定位的文字差异；数学强弱未定"
    if status == "active_text_changed" or _plain(old.get("text")) != _plain(new.get("text")):
        return "声明文字有差异，当前强度规则未归类；强弱未定"
    if old.get("text") and new.get("text"):
        return "给定声明文字相同；声明外变化未查"
    return "声明文本不全，文字差异与强弱均未知"


def _change_summary(algorithm: Mapping, limit: int = 3) -> str:
    """One concrete line naming what changed, for headings and the overview table."""
    if _map(algorithm.get("rewrite_equivalence")).get("suppresses_strength_alert"):
        return "仅改写（限定换写规则已覆盖）"
    premises = [_map(p) for p in _items(algorithm.get("new_premises")) if _map(p)]
    if premises:
        parts = []
        for premise in premises[:limit]:
            label = _plain(premise.get("label") or premise.get("property"))
            label = label.split("：", 1)[-1] if "：" in label else label
            obj = _plain(premise.get("object_key"))
            parts.append((obj + "：" if obj and obj not in label else "") + label)
        more = f" 等 {len(premises)} 项" if len(premises) > limit else ""
        return "新增前提 " + "；".join(_short(part, 80) for part in parts) + more
    if any(c['old_numbers'] != c['new_numbers'] for c in _map(algorithm.get('condition_reference_expansion')).get('comparisons', [])):
        return '引用条件范围变化（列表内容已展开，方向待核）'
    if algorithm.get('unmatched_hypotheses'):
        return '条件从句发生变化（对象或分支对应待核）'
    if any(c.get('status') == 'changed_existing_clause' for c in _items(algorithm.get('hypothesis_comparison'))):
        return '已有前提表述变化' + ('，脚注也有变化' if algorithm.get('footnote_changes') else '') + '（新增限制与否待核）'
    labels = []
    for signal in _main_signals(algorithm):
        label = _plain(_map(signal).get("label"))
        if label and label not in labels:
            labels.append(label)
    if labels:
        return "；".join(_short(label, 80) for label in labels[:2])
    return _plain(algorithm.get("headline")) or "命题对应与变化待核"


def _queue_label(algorithm: Mapping) -> str:
    state = _plain(algorithm.get("candidate_state"))
    def yes_no(value):
        return "是" if value is True else "否" if value is False else "未知"
    return (
        _CANDIDATE_STATES.get(state, state or "队列状态未知")
        + "；进入强度候选队列：" + yes_no(algorithm.get("eligible_for_strength_queue"))
        + "；同篇／同名作者文本修订调查路由：" + yes_no(algorithm.get("eligible_for_self_revision_queue"))
    )


def _role_label(algorithm: Mapping, paper: Mapping, side: str) -> str:
    role = _map(_map(algorithm.get("assertion_role")).get(side))
    if role:
        label = _plain(role.get("label") or role.get("role") or "声明归属未知")
        keys = role.get("citation_keys")
        if keys:
            label += "；引文键 " + _short(keys, 150)
        return label
    if re.search(r"\\cite\w*\*?(?:\[[^\]]*\])?\s*\{", _plain(paper.get("text"))):
        return "声明含引用，所引命题的原创归属未知"
    return "声明片段未见引文；原创归属未核"


def _signal_label(signal: Mapping) -> str:
    label = _plain(signal.get("label") or signal.get("kind") or "未命名规则线索")
    if _plain(signal.get("term")).casefold() == "normal":
        action = signal.get("action")
        prefix = "新增词面：" if action == "added" else "未再出现词面：" if action == "removed" else "词面变化："
        return prefix + "normal（具体术语含义待核）"
    return label


def _signal_location(signal: Mapping) -> str:
    locations = []
    for key, label in (("old_span", "原端"), ("new_span", "对照端")):
        if signal.get(key) is not None:
            locations.append(label + " " + _plain(signal[key]))
    for key, label in (("old_formula_span", "原端公式原子"), ("new_formula_span", "对照端公式原子")):
        if signal.get(key) is not None:
            locations.append(label + " " + _plain(signal[key]))
    if signal.get("changed_span") is not None:
        side = {"old": "原端", "new": "对照端"}.get(signal.get("changed_side"), "端未知")
        locations.append(side + " " + _plain(signal["changed_span"]))
    event = signal.get("event_id") or signal.get("signal_id")
    prefix = ("检测事件 " + _plain(event) + "；") if event else ""
    if not locations:
        return prefix + "未提供位置；不能把重复引文认作独立支持事件"
    unit = _plain(signal.get("span_unit")) or "坐标单位未提供"
    if unit in {"characters_within_displayed_statement", "characters_within_extracted_statement"}:
        unit = "相对展示声明的完整提取基准（裁剪前）；原输入偏移见 statement_slice；不是源码坐标"
    return prefix + "；".join(locations) + "；" + _plain(unit)


def _signal_counts(algorithm: Mapping, displayed: list[Mapping]) -> dict:
    all_signals = [_map(s) for s in _items(algorithm.get("signals")) if _map(s)]
    total_kinds = Counter(_plain(s.get("kind") or "unknown") for s in all_signals)
    shown_kinds = Counter(_plain(s.get("kind") or "unknown") for s in displayed)
    return {"available_total": len(all_signals), "main_available": len(_main_signals(algorithm)),
            "displayed": len(displayed), "omitted_count": max(0, len(all_signals) - len(displayed)),
            "omitted_kinds": dict(total_kinds - shown_kinds),
            "count_unit": "detector_records_not_independent_support_events", "independent_support_event_count": None}


def _summary_signal(signal: Any, include_explanation: bool = True, include_location: bool = True) -> list[str]:
    signal = _map(signal)
    title = _signal_label(signal)
    limit = 800 if signal.get("formula_change_structure") in {"single_to_branches", "branches_to_single"} else 340
    old_quote, new_quote = _paired_quotes(signal.get("old_quote"), signal.get("new_quote"), limit)
    lines = [f"- **{_md(_short(title, 100))}**"]
    if old_quote:
        lines.append(f"  原片段：{_md(old_quote)}")
    if new_quote:
        lines.append(f"  对照片段：{_md(new_quote)}")
    if include_explanation and signal.get("explanation"):
        lines.append(f"  {_md(_short(signal['explanation'], 280))}")
    if include_location:
        if signal.get("scope"):
            lines.append(f"  检查范围：{_md(_short(signal['scope'], 180))}")
        lines.append("  定位：" + _md(_signal_location(signal)))
    return [line + "  " if index < len(lines) - 1 else line for index, line in enumerate(lines)]


def _competition_lines(evidence: Mapping) -> list[str]:
    competition = _map(evidence.get("competition"))
    lines, seen = [], set()
    for side, label in (("forward", "对照端的竞争声明"), ("reverse", "原端的竞争声明")):
        alternative = _map(_map(competition.get(side)).get("best_alternative"))
        if not alternative:
            continue
        pair_id = _plain(alternative.get("pair_id"))
        statement_id = _plain(alternative.get("statement_id"))
        identity = (pair_id, statement_id)
        if identity in seen:
            continue
        seen.add(identity)
        text = label + "：" + _plain(alternative.get("paper_id") or "论文未知")
        if alternative.get("version"):
            text += " v" + _plain(alternative["version"]).lstrip("vV")
        text += "；声明 " + (statement_id or "未知") + "；对号 " + (pair_id or "未知")
        if alternative.get("title"):
            text += "；名称 " + _short(alternative["title"], 150)
        lines.append(_md(text) + "（在 results.jsonl 按标识定位）。")
    return lines


def _focused_source(paper: Mapping, algorithm: Mapping, side: str, context: bool = False) -> str:
    text = _plain(paper.get("context" if context else "text"))
    limit = 480 if context else 900
    needles = []
    if context:
        for raw in _items(_map(algorithm.get("disclosure")).get("matched_quotes")):
            match = _map(raw)
            if match.get("side") == side:
                needles.append(_plain(match.get("quote")))
    else:
        for signal in _main_signals(algorithm):
            needles.append(_plain(signal.get(side + "_quote")))
        for raw in _items(_map(algorithm.get("text_difference")).get("changes")):
            needles.append(_plain(_map(raw).get(side + "_quote")))
    for needle in needles:
        needle = needle.strip("… \n\r")
        if needle:
            start = text.find(needle)
            if start >= 0:
                return _window(text, start + len(needle) // 2, limit)
    return _window(text, 0, limit)


def _compact_source(source: Any) -> dict:
    source = _map(source)
    # The parquet pointer plus statement identifier is sufficient for indexed
    # evidence; repeated manifest/database locations stay in the full record.
    keys = ("path", "sha256", "span", "source_available", "indexed_parquet", "indexed_text_sha256",
            "hydration_status", "text_scope", "context_scope", "input_text_truncated",
            "preview_may_be_truncated", "post_first_end_document", "document_roots", "line_start", "start_line")
    result = {key: source[key] for key in keys if key in source}
    for key in ("path", "sha256", "span", "source_available"):
        result.setdefault(key, None)
    result["source_access"] = _source_access_summary(source)
    return result


def _compact_changes(difference: Mapping) -> dict:
    changes = []
    available = [_map(c) for c in _items(difference.get("changes")) if _map(c)]
    for change in available[:8]:
        row = {key: change[key] for key in ("kind", "event_id", "old_span", "new_span", "span_unit", "quote_source") if key in change}
        has_minimum = "old_changed_text" in change or "new_changed_text" in change
        if has_minimum:
            for side in ("old", "new"):
                value = _plain(change.get(side + "_changed_text"))
                row[side + "_changed_text"] = value[:500]
                total = change.get(side + "_changed_text_total_characters")
                upstream_truncated = change.get(side + "_changed_text_truncated")
                displayed = len(row[side + "_changed_text"])
                row[side + "_changed_text_total_characters"] = total
                row[side + "_changed_text_displayed_characters"] = displayed
                if len(value) > 500 or upstream_truncated is True or (isinstance(total, int) and total > displayed):
                    truncated = True
                else:
                    truncated = False if upstream_truncated is False else None
                row[side + "_changed_text_truncated"] = truncated
        else:
            row["old_quote"], row["new_quote"] = _paired_quotes(change.get("old_quote"), change.get("new_quote"), 300)
        changes.append(row)
    return {"status": difference.get("status") or "unknown", "logical_strength": difference.get("logical_strength") or "unknown",
            "coordinate_system": difference.get("coordinate_system"),
            "comparison_scope": difference.get("comparison_scope"), "changes": changes,
            "changes_displayed": len(changes), "changes_available_in_full_record": len(available),
            "engine_changes_display_limit": difference.get("changes_display_limit"),
            "all_mathematical_changes_covered": False}


def summarize_result(item: Mapping) -> dict:
    """Compact evidence ledger; omissions and unknown states remain explicit."""
    item = _map(item)
    algorithm = _map(item.get("algorithm"))
    evidence = _map(algorithm.get("alignment_evidence"))
    support = _map(item.get("support"))
    compact = {"pair_id": item.get("pair_id"), "schema_version": "compact-user-evidence-v2", "channels": item.get("channels", [])}
    for side in ("old", "new"):
        paper = _map(item.get(side))
        compact[side] = {key: paper.get(key) for key in ("paper_id", "statement_id", "version", "date", "date_kind", "date_source", "first_submission_date", "version_date", "version_date_basis", "authors", "statement_reference_keys", "statement_reference_basis")}
        compact[side]["source"] = _compact_source(paper.get("source"))
        for key in ("statement_slice", "statement_coordinates", "input_text_length", "display_text_truncated", "text_derivation"):
            compact[side][key] = paper.get(key)
        compact[side]["result_name"] = evidence.get(side + "_title") or _result_name(paper) or None
        role = dict(_map(_map(algorithm.get("assertion_role")).get(side)))
        if not role:
            role = {"label": _role_label(algorithm, paper, side), "ownership_status": "unknown"}
        role.pop("paper_authors_are_not_assertion_owners_by_default", None)
        role["matched_quotes"] = [{**_map(q), "quote": _short(_map(q).get("quote"), 320)} for q in _items(role.get("matched_quotes"))[:3]]
        compact[side]["assertion_role"] = role
    main = _main_signals(algorithm)
    displayed = main[:3]
    compact_signals, notes = [], {}
    note_ids = {}
    for signal in displayed:
        row = {key: signal[key] for key in (
            "kind", "scope", "term", "action", "direction_hint", "logical_role_status", "quote_source",
            "signal_id", "event_id", "old_span", "new_span", "old_formula_span", "new_formula_span", "changed_span", "changed_side", "span_unit",
            "old_comparison_object", "new_comparison_object", "old_branch_condition", "new_branch_condition",
            "asymptotic_subject", "asymptotic_limit", "old_order", "new_order",
            "formula_change_structure", "parameter_domain_candidate") if key in signal}
        row["label"] = _signal_label(signal)
        limit = 800 if signal.get("formula_change_structure") in {"single_to_branches", "branches_to_single"} else 340
        row["old_quote"], row["new_quote"] = _paired_quotes(signal.get("old_quote"), signal.get("new_quote"), limit)
        explanation = _plain(signal.get("explanation"))
        if explanation:
            key = note_ids.setdefault(explanation, "note_" + str(len(note_ids) + 1))
            notes[key] = explanation
            row["explanation_ref"] = "algorithm.signal_notes." + key
        if not any(signal.get(key) is not None for key in ("old_span", "new_span", "changed_span")):
            row["independent_occurrence_status"] = "unknown_without_location"
        compact_signals.append(row)
    disclosure = _map(algorithm.get("disclosure"))
    chronology = _map(algorithm.get("chronology"))
    authors = _map(algorithm.get("author_relation"))
    next_step = _map(algorithm.get("next_step"))
    basis = _map(algorithm.get("self_revision_basis"))
    compact["algorithm"] = {
        "version": algorithm.get("version"), "statement_alignment": algorithm.get("statement_alignment"),
        "new_premises": algorithm.get("new_premises", []),
        "hypothesis_comparison": algorithm.get("hypothesis_comparison", []),
        "reference_expansion": algorithm.get("reference_expansion", {}),
        "footnote_changes": algorithm.get("footnote_changes", []),
        "reading_group": algorithm.get("reading_group", {}),
        "unmatched_hypotheses": algorithm.get("unmatched_hypotheses", {}),
        "condition_reference_expansion": algorithm.get("condition_reference_expansion", {}),
        "change_analysis": algorithm.get("change_analysis"),
        "formula_scope_analysis": algorithm.get("formula_scope_analysis"),
        "scope_review_required": algorithm.get("scope_review_required"),
        "scope_evidence_behind_gate": algorithm.get("scope_evidence_behind_gate"),
        "scope_review_gate_note": algorithm.get("scope_review_gate_note"),
        "rewrite_equivalence": algorithm.get("rewrite_equivalence"),
        "tightening_textual_evidence": algorithm.get("tightening_textual_evidence"),
        "disclosure_evidence": _compact_disclosure_evidence(algorithm.get("disclosure_evidence")),
        "investigation_queue": algorithm.get("investigation_queue"),
        "priority_reasons": algorithm.get("priority_reasons", []),
        "eligible_for_primary_investigation_queue": algorithm.get("eligible_for_primary_investigation_queue"),
        "candidate_state": algorithm.get("candidate_state"), "eligible_for_strength_queue": algorithm.get("eligible_for_strength_queue"), "eligible_for_self_revision_queue": algorithm.get("eligible_for_self_revision_queue"),
        "self_revision_basis": {key: basis.get(key) for key in ("status", "passed", "pending")},
        "alignment_label": _alignment_label(algorithm), "decision_status": algorithm.get("decision_status"),
        "alignment_evidence": {"status": evidence.get("status"), "common_anchors": _items(evidence.get("common_anchors"))[:8], "old_conclusion_excerpt": _short(evidence.get("old_conclusion_excerpt"), 300), "new_conclusion_excerpt": _short(evidence.get("new_conclusion_excerpt"), 300), "competition": evidence.get("competition")},
        "chronology": {key: chronology.get(key) for key in ("status", "label", "direction", "basis", "same_paper", "dates_are_statement_snapshot_dates", "statement_order_confirmed", "statement_snapshot_date_status", "statement_snapshot_dates")},
        "author_relation": {key: authors.get(key) for key in ("kind", "label", "matched_names", "identity_verified", "basis", "historical_version_author_lists_verified")},
        "text_difference_label": _difference_label(algorithm, _map(item.get("old")), _map(item.get("new"))),
        "text_difference": _compact_changes(_map(algorithm.get("text_difference"))),
        "signals": compact_signals, "signal_notes": notes, "signal_coverage": _signal_counts(algorithm, displayed),
        "additional_main_signal_count": max(0, len(main) - len(displayed)),
        "priority_score": algorithm.get("priority_score"), "priority_band": algorithm.get("priority_band"),
        "important_gap": _important_gap(algorithm),
        "next_step": {key: next_step.get(key) for key in ("action", "label", "reason", "automatic", "executed", "selection_basis")} if next_step else {"label": _next_step(algorithm)},
        "disclosure": {"status": disclosure.get("status"), "label": disclosure.get("label") or _DISCLOSURE.get(_plain(disclosure.get("status"))), "coverage": disclosure.get("coverage"), "matched_quotes": [{"side": _map(q).get("side"), "scope": _map(q).get("scope"), "quote": _short(_map(q).get("quote"), 360)} for q in _items(disclosure.get("matched_quotes"))[:2]]},
        "weaker_claim_certified": algorithm.get("weaker_claim_certified"),
        "mathematical_entailment_checked": algorithm.get("mathematical_entailment_checked"),
    }
    compact["support"] = {key: support.get(key) for key in (
        "group_key", "raw_observed_distinct_later_papers", "raw_observed_later_paper_ids",
        "qualified_distinct_later_papers", "qualified_later_paper_ids", "verified_adoption", "verified_adoption_status",
        "distinct_later_papers", "same_author_name_later_papers", "third_party_name_later_papers", "papers_without_author_metadata", "distinct_author_groups", "same_author_name_later_paper_ids", "third_party_name_later_paper_ids", "distinct_version_observations", "later_paper_ids")}
    compact["support"].update(add_across_rows=False, groups_file="groups.jsonl", count_scope="本批观察与通过门控分列；确认沿用未核，不跨行相加")
    compact["export"] = {**_map(item.get("export")), "full_record_file": "results.jsonl", "full_record_key": item.get("pair_id"), "summary_is_not_a_complete_source_record": True}
    compact["evidence_documents"] = [_compact_document(doc, _items(_map(algorithm.get("disclosure_evidence")).get("matched_evidence"))) for doc in _documents(item)]
    return compact


def _render_item(item: Mapping, number: int) -> list[str]:
    algorithm = _map(item.get("algorithm"))
    old, new = _map(item.get("old")), _map(item.get("new"))
    evidence = _map(algorithm.get("alignment_evidence"))
    chronology = _map(algorithm.get("chronology"))
    authors = _map(algorithm.get("author_relation"))
    disclosure = _map(algorithm.get("disclosure"))
    support = _map(item.get("support"))
    queue = _map(algorithm.get("investigation_queue"))
    tightening = _map(algorithm.get("tightening_textual_evidence"))
    rewriting = _map(algorithm.get("rewrite_equivalence"))
    old_name = _plain(evidence.get("old_title")) or _result_name(old)
    new_name = _plain(evidence.get("new_title")) or _result_name(new)
    if old_name and new_name:
        headline = _change_summary(algorithm) + "（" + old_name + " → " + new_name + "）"
    elif algorithm.get("priority_band") == "no_signal":
        headline = _difference_label(algorithm, old, new)
    else:
        headline = _change_summary(algorithm)
    pair_id = item.get("pair_id") or f"无标识-{number}"
    lines = [f"## {number}. {_md(_short(headline, 190))}", ""]
    diagnostics = [
        "**算法去向：** " + _md(_status_label(queue) if queue else _queue_label(algorithm)) + "。", "",
        "**命题对应：** " + _md(_alignment_label(algorithm)) + "。", "",
        "**首要缺口：** " + _md(_important_gap(algorithm).rstrip("。．. ")) + "。",
        "**下一步核验：** " + _md(_short(_next_step(algorithm), 260).rstrip("。．. ")) + "。", "",
    ]
    band = _plain(algorithm.get("priority_band"))
    diagnostics.extend([f"调查排序：{_md(_BANDS.get(band, band or '未分级'))} · 启发优先分 {_score_label(item)} · 对号 {_md(pair_id)}", ""])
    lines.extend([
        "| 信息 | 原陈述端 | 对照端 |", "|---|---|---|",
        f"| 结果名称（可见原文） | {_md(old_name or '未取得声明名称')} | {_md(new_name or '未取得声明名称')} |",
        f"| 论文／版本 | {_paper(old)} | {_paper(new)} |",
        f"| 实际版本日期 | {_version_date(old)} | {_version_date(new)} |",
        f"| 论文作者姓名 | {_authors(old)} | {_authors(new)} |",
        f"| 声明角色／归属 | {_md(_role_label(algorithm, old, 'old'))} | {_md(_role_label(algorithm, new, 'new'))} |",
        f"| 静态编号线索（非 PDF 认证） | {_md(_short(old.get('statement_reference_keys'), 180) or '未知')} | {_md(_short(new.get('statement_reference_keys'), 180) or '未知')} |",
        f"| 完整 TeX／声明位置 | {_full_source_ref(old.get('source'))} | {_full_source_ref(new.get('source'))} |",
    ])
    detail = diagnostics
    if rewriting.get("suppresses_strength_alert"):
        detail.extend(["**程序识别的变化：** " + _md(_status_label(rewriting)) + "。限定换写规则已覆盖这次差异，不再用原始词元比较发出混合强度提示；规则之外的数学等价仍未认证。", ""])
    elif tightening:
        detail.extend(["**程序识别的变化：** " + _md(_status_label(tightening)) + "；结论对应（文本口径）：" + _md(_status_label(tightening.get("conclusion_match"))) + "。", ""])
    if evidence.get("old_conclusion_excerpt") or evidence.get("new_conclusion_excerpt"):
        missing_conclusion='未提取出结论片段；不能据此推断删去结论'
        lines.append(f"| 结论片段 | {_md(_short(evidence.get('old_conclusion_excerpt'), 240) or missing_conclusion)} | {_md(_short(evidence.get('new_conclusion_excerpt'), 240) or missing_conclusion)} |")
    lines.append("")
    if rewriting.get('suppresses_strength_alert') or tightening:
        lines.extend(["**说明状态：** " + _md(_core_disclosure_label(algorithm)) + "。", ""])
    reading = _map(algorithm.get('reading_group'))
    if reading.get('repeated_text'):
        lines.extend([f"**重复表述组：** {reading.get('pair_count')} 对共享相同版本间的前后文本，仅作1个阅读组，不能算多个独立发现。", "",
                      "组内对号：" + _md(reading.get('member_pair_ids')) + "。", ""])
    clauses = [c for c in _items(algorithm.get('hypothesis_comparison')) if c.get('status') != 'same_main_clause']
    if clauses:
        from premise_clauses import masks
        lines.extend(['**已对应的条件从句（主句正文，脚注另列）：**', '', '| 改前 | 改后 |', '|---|---|'])
        for c in clauses[:6]:
            a = re.sub(r'\s+', ' ', masks(c['old_quote'])[1]).strip()
            b = re.sub(r'\s+', ' ', masks(c['new_quote'])[1]).strip()
            lines.append('| ' + _md(a) + ' | ' + _md(b) + ' |')
        lines.extend(['', '同一主语已有前提发生变化；未自动认作全新前提。此表不覆盖整条定理的全部条件；原始从句及坐标保存在 JSON。', ''])
    unpaired = _map(algorithm.get('unmatched_hypotheses'))
    if unpaired:
        from premise_clauses import masks
        left = '<br>'.join(_md(re.sub(r'\s+', ' ',masks(c['quote'])[1]).strip()) for c in unpaired['old']) or '未提取到对应条件从句；不等于不存在'
        right = '<br>'.join(_md(re.sub(r'\s+', ' ',masks(c['quote'])[1]).strip()) for c in unpaired['new']) or '未提取到对应条件从句；不等于不存在'
        lines.extend(['**其余条件从句的变化（尚未建立逐项对应）：**', '', '| 原端条件集 | 对照端条件集 |', '|---|---|',
                      '| '+left+' | '+right+' |', '', '这里只并列提取结果，不把未匹配直接当新增或删除，也不推断分支间的蕴含方向。', ''])
    for c in _map(algorithm.get('condition_reference_expansion')).get('comparisons', []):
        lines.extend(['**条件列表展开：** '+_md(c['label'])+'；原引用 '+_md(c['old_numbers'])+'，新引用 '+_md(c['new_numbers'])+'。', '',
                      '| 新编号 | 对应线索 | 新条件正文 |', '|---|---|---|'])
        for row in c['mapping']:
            if 'old_number' in row:
                reason='旧第'+str(row['old_number'])+'条排版归一化后字面相同；'+('旧声明已引用' if row['old_was_selected'] else '旧声明未列入该编号范围')
            else:
                possible=row.get('possible_text_matches', [])
                reason='没有唯一字面对应'
                if possible: reason+='；可能对应旧第'+str(possible[0]['old_number'])+'条（仅文字相似，未认证）'
            lines.append('| '+str(row['new_number'])+' | '+_md(reason)+' | '+_md(row['new_item']['quote'])+' |')
        lines.extend(['', '按内容比较，不按编号数相减推算新增前提。候选对应及旧条目全文见 JSON；宏含义、旧引用外条件是否另有全局约定尚未核验。', ''])
    refs = _map(algorithm.get('reference_expansion'))
    relevant_labels = set()
    for clause in clauses:
        relevant_labels.update(re.findall(r'\\(?:eqref|[cC]ref|[rR]ef)\s*\{([^{}]+)\}', clause['old_quote'] + clause['new_quote']))
    for c in _items(refs.get('comparisons'))[:8]:
        status = '对应公式字面相同' if c['status'] == 'same_literal_equation' else '引用目标公式发生变化'
        destination = lines if c['label'] in relevant_labels else detail
        destination.extend(['**引用展开：** ' + _md(c['label']) + ' — ' + status + '。', '',
                      '| 原端目标 | 对照端目标 |', '|---|---|',
                      '| ' + _md(c['old']['quote']) + ' | ' + _md(c['new']['quote']) + ' |', '',
                      '目标在所给证据文档 ' + _md(c['old'].get('doc_id')) + ' 第 ' + str(c['old']['start_line']) + ' 行／' + _md(c['new'].get('doc_id')) + ' 第 ' + str(c['new']['start_line']) + ' 行；这些是输入证据坐标，不自动等于原库阅读器行号。该比较不认证整个前提等价。', ''])
    unresolved = {s: [k for k, v in _map(_map(refs.get('endpoints')).get(s)).items() if v.get('status') == 'unresolved_or_ambiguous'] for s in ('old', 'new')}
    if any(unresolved.values()):
        detail.extend(['未展开或有歧义的引用／标签：' + _md(unresolved) + '；未由公式或静态编号条件列表规则解析，保留待核。', ''])
    for notes in _items(algorithm.get('footnote_changes')):
        lines.extend(['**脚注也有变化：** 可能涉及定义或范围，未混入主句，也未当作纯改写删除。', ''])
        detail.extend(['**脚注全文对照：**', '', '| 改前脚注 | 改后脚注 |', '|---|---|',
                       '| ' + _md('\n'.join(notes['old_notes']) or '无') + ' | ' + _md('\n'.join(notes['new_notes']) or '无') + ' |', ''])
    competition_lines = _competition_lines(evidence)
    if competition_lines:
        detail.extend(["**竞争候选入口（当前输入范围）：**", "", *competition_lines, ""])
    relation_label = authors.get("label") or authors.get("kind") or "作者姓名关系未知"
    chronology_label = chronology.get("label") or chronology.get("status") or "先后方向未知"
    lines.extend([f"**姓名关系：** {_md(relation_label)}。 **时间方向：** {_md(chronology_label)}。", ""])
    if chronology.get("status") == "paper_date_only":
        lines.extend(["只确认论文首发顺序，尚未确认这两段声明的版本先后。", ""])
    elif not chronology.get("statement_order_confirmed"):
        lines.extend(["两段声明先后未确认；左右排列不代表修订方向。", ""])
    if evidence.get("common_anchors"):
        detail.extend(["共同文字锚点：" + _md(_short(evidence["common_anchors"], 180)) + "。", ""])
    detail.extend(["**文字检测：** " + _md(_difference_label(algorithm, old, new)) + "。", ""])
    lines.extend(_premise_lines(algorithm))
    analysis = _map(algorithm.get('change_analysis'))
    if analysis and not rewriting.get('suppresses_strength_alert'):
        lines.extend(['**具体变化类型：** '+_md(analysis.get('label'))+'。',''])
        scope_findings = [_map(f) for f in _items(analysis.get('findings'))]
        for finding in scope_findings[:5]:
            role = {'premise':'前提位置','declaration':'参数声明位置','definition_or_fixed_parameter':'定义或固定参数位置','universal_domain':'全称量词范围','postposed_domain':'后置适用范围',
                    'conclusion':'结论位置','branch_domain':'分段公式的局部条件','unscoped_formula':'公式角色尚未定位',
                    'biconditional_operand':'当且仅当的一端','existential_operand':'存在量词的一端',
                    'nonconjunctive_scope':'含析取，未按条件合取处理'}.get(finding.get('syntactic_role'),'声明范围')
            lines.extend(['- '+_md(finding.get('label'))+'；'+_md(role)+'。',
                '  原端：'+_code(finding.get('old_quote') or '本声明中未找到对应公式')+'；对照端：'+_code(finding.get('new_quote') or '本声明中未再出现对应公式')+'。',
                '  规则：'+_md(finding.get('comparison_rule'))+'。'])
        if len(scope_findings)>5:
            lines.append('另有 '+str(len(scope_findings)-5)+' 项范围记录保留在 JSON。')
        lines.append('')
        ledger=_map(_map(algorithm.get('formula_scope_analysis')).get('formula_edit_ledger'))
        for key,name in [('exponents','指数'),('subscripts','下标'),('macros','宏/命令')]:
            entry=_map(ledger.get(key))
            if entry.get('multiset_changed'):
                old_values='；'.join(_plain(_map(v).get('text'))+' ×'+_plain(_map(v).get('count')) for v in _items(entry.get('removed'))[:5])
                new_values='；'.join(_plain(_map(v).get('text'))+' ×'+_plain(_map(v).get('count')) for v in _items(entry.get('added'))[:5])
                detail.append('公式'+name+'词元账目：减少 '+_code(old_values or '无')+'；增加 '+_code(new_values or '无')+'。')
        detail.append('')
    if rewriting:
        filtered = "；该改写按规则排除出强度警报" if rewriting.get("suppresses_strength_alert") is True else ""
        detail.extend(["**改写过滤：** " + _md(_status_label(rewriting)) + _md(filtered) + "。", ""])
    signals = _main_signals(algorithm)
    coverage = _signal_counts(algorithm, signals[:3])
    detail.extend([f"检测记录展示 {coverage['displayed']}/{coverage['available_total']} 条；省略类型：" + _md(coverage["omitted_kinds"] or "无") + "。记录数不是独立支持事件数。", ""])
    for index, change in enumerate(_compact_changes(_map(algorithm.get("text_difference")))["changes"], 1):
        for side, label in (("old", "原端"), ("new", "对照端")):
            if change.get(side + "_changed_text_truncated"):
                total = _count(change.get(side + "_changed_text_total_characters"))
                displayed = _count(change.get(side + "_changed_text_displayed_characters"))
                detail.extend([f"JSON 最小差异账目第 {index} 处{label}：原变化 {total} 字符，保留 {displayed} 字符，**已截断**。", ""])
    # Premise additions already have their own table above; list the other changes.
    top_signals = [s for s in signals if not (algorithm.get("new_premises") and s.get("kind") == "aligned_premise_addition")]
    if top_signals:
        lines.extend(["**具体差异：**", ""])
        seen_explanations = set()
        for signal in top_signals[:3]:
            explanation = _plain(signal.get("explanation"))
            lines.extend(_summary_signal(signal, include_explanation=explanation not in seen_explanations, include_location=False))
            seen_explanations.add(explanation)
            lines.append("")
        if len(top_signals) > 3:
            lines.extend([f"另有 {len(top_signals) - 3} 条主要线索保留在完整 JSON。", ""])
    if signals:
        detail.extend(["**差异定位：**", ""])
        for signal in signals[:3]:
            detail.append("- " + _md(_short(_signal_label(signal), 100)) + "：检查范围 " + _md(_short(signal.get("scope") or "未知", 120)) + "；" + _md(_signal_location(signal)))
        detail.append("")
    else:
        changes = _items(_map(algorithm.get("text_difference")).get("changes"))
        if not rewriting.get('suppresses_strength_alert') and _map(algorithm.get("text_difference")).get("status") != "formatting_only":
            for raw in changes[:2]:
                change = _map(raw)
                lines.extend(_summary_signal({**change, "label": "未归类的可见文字差异；数学作用待核"}))
                lines.append("")
    usage = [s for s in _items(algorithm.get("signals")) if _map(s).get("kind") in _USAGE_SIGNALS]
    if usage:
        lines.extend(["**应用／引用语境：** " + _md(_short(_map(usage[0]).get("new_quote"), 230)) + "（尚未确认所引对象就是原陈述）。", ""])
    raw_count = support.get("raw_observed_distinct_later_papers")
    qualified_count = support.get("qualified_distinct_later_papers", support.get("distinct_later_papers"))
    adoption = support.get("verified_adoption")
    adoption_label = "未核实（不是零篇）" if adoption is None else _plain(adoption)
    detail.extend([
        f"**本批后文候选观察：** 观察 {_count(raw_count)} 篇 → 通过候选门控 {_count(qualified_count)} 篇；确认沿用：{_md(adoption_label)}。",
        f"合格项中姓名重合 {_count(support.get('same_author_name_later_papers'))} 篇、姓名不重合 {_count(support.get('third_party_name_later_papers'))} 篇、姓名资料不全 {_count(support.get('papers_without_author_metadata'))} 篇；另有 {_count(support.get('distinct_version_observations'))} 次版本观察。",
        "候选组 " + _md(support.get("group_key") or "未知") + "：同组统计在各行重复引用，**不可跨行相加**；未验证实际沿用。", "",
    ])
    if support.get("raw_observed_later_paper_ids"):
        detail.append("观察到的后文成员：" + _md(_short(support["raw_observed_later_paper_ids"], 360)) + "。")
    basis = _map(algorithm.get("self_revision_basis"))
    if basis:
        detail.append("文本修订调查路由：已具备 " + _md(_short(basis.get("passed") or "未提供", 180)) + "；仍待核 " + _md(_short(basis.get("pending") or "未提供", 180)) + "。原创命题归属另看独立状态。")
    for field, label in (("same_author_name_later_paper_ids", "姓名重合后文"), ("third_party_name_later_paper_ids", "姓名不重合后文")):
        members = _items(support.get(field))
        if members:
            detail.append(label + "：" + _md(members[:12]) + (f"；另 {len(members) - 12} 篇见 groups.jsonl" if len(members) > 12 else "") + "。")
    author_groups = _items(support.get("author_groups"))
    for raw in author_groups[:5]:
        group = _map(raw)
        group_id = group.get("group_id") or group.get("id") or "组标识未知"
        papers = group.get("paper_ids") or group.get("member_paper_ids") or group.get("papers") or []
        names = group.get("normalized_names") or group.get("names") or []
        detail.append("姓名连通组 " + _md(group_id) + "：论文 " + _md(_short(papers, 210)) + "；姓名 " + _md(_short(names, 160)) + "。")
    if len(author_groups) > 5:
        detail.append(f"另 {len(author_groups) - 5} 个姓名连通组见 groups.jsonl。")
    if not author_groups and support.get("later_paper_ids"):
        detail.append("组内后文成员：" + _md(_short(support["later_paper_ids"], 360)) + "；姓名组成员明细未提供。")
    detail.append("")
    status = _plain(disclosure.get("status"))
    disclosure_label = _core_disclosure_label(algorithm) if algorithm.get("disclosure_evidence") else disclosure.get("label") or _DISCLOSURE.get(status) or status or "说明筛查状态未知"
    if not (rewriting.get("suppresses_strength_alert") or tightening) or disclosure_label != _core_disclosure_label(algorithm):
        lines.extend(["**说明筛查：** " + _md(disclosure_label) + "。", ""])
    if algorithm.get("disclosure_evidence"):
        scopes = _map(algorithm.get("disclosure_evidence")).get("statement_notice_strength_scopes")
        if scopes:
            lines.extend(["说明涉及范围（程序线索）：" + _md(_short(scopes, 420)) + "。", ""])
        lines.extend(_evidence_lines(item))
    lines.extend([
        "**可核验位置：** 原陈述 " + _location(old.get("source")) + "；对照 " + _location(new.get("source")) + "。", "",
        "<details>", "<summary>对齐依据、后文观察、差异窗口与来源记录</summary>", "",
        *detail,
        "下方只是裁剪窗口，不是定位坐标的起点。坐标相对展示声明的完整提取基准（裁剪前，Unicode 码点，零起点半开区间）；原输入偏移见 statement_slice；不是源码坐标。", "",
    ])
    for side, side_label, paper in (("old", "原陈述端", old), ("new", "对照端", new)):
        lines.extend(["**" + side_label + "声明窗口**", "", _code(_focused_source(paper, algorithm, side) or "未提供声明文本"), ""])
        if paper.get("context"):
            lines.extend(["给定局部语境窗口：", "", _code(_focused_source(paper, algorithm, side, context=True)), ""])
        if paper.get("statement_slice"):
            lines.extend(["声明切片：" + _md(paper["statement_slice"]) + "（输入文本内坐标，与源码范围分开）。", ""])
        if paper.get("statement_coordinates"):
            lines.extend(["声明坐标说明：" + _md(paper["statement_coordinates"]), ""])
        if paper.get("statement_reference_basis"):
            lines.extend(["静态声明编号依据：" + _md(_short(paper["statement_reference_basis"], 380)) + "（不等同于执行 TeX 或核验 PDF 编号）。", ""])
        source = _map(paper.get("source"))
        if source.get("sha256"):
            lines.extend(["来源 SHA-256：" + _md(source["sha256"]), ""])
        access = _source_access_summary(source)
        if access.get("source_scope") or access.get("statement_match_scope"):
            lines.extend(["完整文件范围：" + _md(access.get("source_scope") or "未知") + "；声明匹配范围：" + _md(access.get("statement_match_scope") or "未知") + "。", ""])
        if access.get("sha256"):
            lines.extend(["完整文件 SHA-256：" + _md(access["sha256"]), ""])
        if access.get("project_root"):
            lines.extend([_file_link(access["project_root"], "打开源码项目目录"), ""])
        for root in access.get("related_roots", [])[:3]:
            root_path = _map(root).get("path") if isinstance(root, Mapping) else root
            if root_path:
                lines.append("关联 TeX 根：" + _file_link(root_path, "打开关联根文件"))
        if access.get("related_roots"):
            lines.append("")
    for raw in _items(disclosure.get("matched_quotes"))[:2]:
        quote = _map(raw)
        lines.extend(["命中说明片段（" + _md(quote.get("side") or "端未知") + "，" + _md(quote.get("scope") or "范围未知") + "）：" + _md(_short(quote.get("quote"), 360)), ""])
    warnings = [_plain(w) for w in _items(algorithm.get("warnings")) if _plain(w)]
    extra = [warning for warning in warnings if warning != _important_gap(algorithm)]
    if extra:
        lines.extend(["其他缺口：" + "；".join(_md(_short(w, 170)) for w in extra[:3]), ""])
    reason_records = algorithm.get("priority_reasons") or algorithm.get("score_components")
    reasons = [_map(c).get("reason") for c in _items(reason_records) if _map(c).get("reason")]
    if reasons:
        lines.extend(["排序依据：" + "；".join(_md(_short(r, 120)) for r in reasons[:4]), ""])
    if item.get("channels"):
        lines.extend(["发现通道：" + _md(item["channels"]), ""])
    lines.extend(["</details>", "", "---", ""])
    return lines


def render_markdown(results: list, metadata: Mapping | None = None) -> str:
    """Render a deterministic ranked preview, preserving full data in JSONL.

    ``metadata.max_display`` defaults to 30 and is bounded at 200.  Sorting is
    by the supplied core queue rank, then descending heuristic score and pair
    identifier, unless ``preserve_input_order`` is requested. Rendering neither
    filters nor changes the full engine result file.
    """
    metadata = _map(metadata)
    raw = list(results or [])
    items = [_map(value) for value in raw]
    if metadata.get("preserve_input_order") is True:
        order_label = "保留调用方传入的队列顺序"
    elif any(math.isfinite(_core_rank(item)) for item in items):
        items.sort(key=lambda item: (_core_rank(item), -_score(item), _plain(item.get("pair_id"))))
        order_label = "按核心调查队列 rank 升序、启发优先分降序、对号升序"
    else:
        items.sort(key=lambda item: (-_score(item), _plain(item.get("pair_id"))))
        order_label = "按启发优先分降序、对号升序"
    try:
        max_display = min(200, max(1, int(metadata.get("max_display", DEFAULT_MAX_DISPLAY))))
    except (TypeError, ValueError):
        max_display = DEFAULT_MAX_DISPLAY
    reading_items, seen_groups = [], set()
    for item in items:
        group = _map(_map(item.get('algorithm')).get('reading_group'))
        key = group.get('group_id') or item.get('pair_id')
        if key in seen_groups: continue
        seen_groups.add(key)
        reading_items.append(item)
    shown = reading_items[:max_display]
    statuses = Counter(_plain(_map(_map(item.get("algorithm")).get("chronology")).get("status")) for item in items)
    exact = statuses["version_order"] + statuses["version_date_order"]
    paper_only = statuses["paper_date_only"]
    unresolved = len(items) - exact - paper_only
    title = metadata.get("title") or "命题强度变化线索"
    lines = [
        "# " + _md(title),
        "",
        f"共 {len(items)} 条结果，合并重复表述后 {len(reading_items)} 个阅读组；{order_label}展示前 {len(shown)} 组。完整记录见 results.jsonl；精简机器读物见 summary.jsonl；去重候选组见 groups.jsonl。",
        "",
        "这是旧候选的复查集合；不能用剩余条数、入列率或未入列率估计全库召回。精简 JSON 只含短证据和完整源码指针，未包含整份 TeX。",
        "",
        f"**时间范围：** {exact} 条可确认版本／版本日期顺序，{paper_only} 条仅知论文首发先后，{unresolved} 条方向待核或有冲突。",
        "",
        "**读法：** 局部条件的收紧或放宽，不等于整条命题已认证弱化；分数不是准确率。说明结论只限已扫描材料。先看前后对照，再按全文入口核查。",
        "",
    ]
    for field, label in (("scope", "本次输入范围"), ("input_description", "输入说明")):
        if metadata.get(field):
            lines.extend(["**" + label + "：** " + _md(_short(metadata[field], 700)), ""])
    if any(_map(item.get("algorithm")).get("investigation_queue") for item in items):
        counts = summary_counts(items)
        queue_counts = {_INVESTIGATION_QUEUES.get(key, key): count for key, count in counts["investigation_queues"].items()}
        disclosure_counts = {_DISCLOSURE_EVIDENCE.get(key, key): count for key, count in counts["disclosure_states"].items()}
        lines.extend(["**核心队列计数：** " + _md(queue_counts) + "。", "",
                      "**说明状态计数：** " + _md(disclosure_counts) + "。", ""])
    if metadata.get("limit_applied"):
        lines.extend([f"输入按上限截取 {_md(metadata.get('selected_pairs', '未知'))} 对，另有 {_md(metadata.get('excluded_due_to_limit', '未知'))} 行未分析；不代表全输入排序。", ""])
    if not items:
        lines.extend(["本次没有可展示的结果。空结果不表示没有强度变化，也不提供漏检率估计。", ""])
    elif len(items) > len(shown):
        lines.extend(["未展示项保留在完整 JSONL 中；本页截断仅影响阅读，不删除候选。", ""])
    if shown:
        lines.extend(["## 总览", "", "| # | 对号 | 排序 | 程序识别的变化 | 说明状态 |", "|---|---|---|---|---|"])
        for number, item in enumerate(shown, 1):
            algorithm = _map(item.get("algorithm"))
            band = _plain(algorithm.get("priority_band"))
            lines.append(f"| {number} | {_md(item.get('pair_id') or '无标识')} | {_md(_BANDS.get(band, band or '未分级'))} · {_score_label(item)} | "
                         f"{_md(_short(_change_summary(algorithm), 160))} | {_md(_short(_core_disclosure_label(algorithm), 60))} |")
        lines.append("")
    for number, item in enumerate(shown, 1):
        lines.extend(_render_item(item, number))
        group = _map(_map(item.get('algorithm')).get('reading_group'))
        peers = [r for r in items if r.get('pair_id') != item.get('pair_id') and r.get('pair_id') in group.get('member_pair_ids', [])]
        if peers:
            lines.extend(['**同组其他出处（保留全部定位，不重复展示相同正文）：**', ''])
            for peer in peers:
                lines.append('- ' + _md(peer.get('pair_id')) + '：原端 ' + _full_source_ref(_map(peer.get('old')).get('source')) + '；对照端 ' + _full_source_ref(_map(peer.get('new')).get('source')))
            lines.append('')
    return "\n".join(lines).rstrip() + "\n"
