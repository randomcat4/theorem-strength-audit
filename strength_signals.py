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
import re
import unicodedata
from typing import Iterable
from formula_scope import analyze_formula_scope, added_formula_premises, classify_change, condition_rewrite_projection

VERSION = "0.5.0"
__version__ = VERSION

MAX_TEXT = 24000
MAX_CONTEXT = 12000
MAX_DISPLAY_TEXT = 9000
MAX_TOKENS = 5000
MAX_EVIDENCE_DOCUMENTS = 12
MAX_EVIDENCE_TEXT = 1000000
MAX_EVIDENCE_CACHE = 64

_DECL = r"theorem|thm|lemma|lem|proposition|prop|corollary|cor|claim|conjecture|conj|definition|defn|remark|rem"
_BEGIN_DECL = re.compile(r"\\begin\s*\{(" + _DECL + r")\*?\}", re.I)
_PROOF = re.compile(r"\\begin\s*\{proof\*?\}|(?:^|\n)\s*Proof\s*[.:]", re.I)
_META = re.compile(r"\\(?:label|(?:[cCvV]?[cC])?ref|eqref|cite\w*|tag)\*?(?:\[[^\]]*\])?\s*\{[^{}]*\}")
_TOKEN = re.compile(r"\\[A-Za-z]+|[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?|[\u3400-\u9fff]|<=|>=|!=|[≤≥≠∀∃¬<>=+*/^_{}()\[\]!|&-]")
_FORMULA = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", re.S)
_MATH_BLOCK = re.compile(
    r"\$\$(?P<dollar_display>.+?)\$\$|\\\[(?P<bracket_display>.+?)\\\]|"
    r"\\\((?P<paren_inline>.+?)\\\)|(?<!\\)\$(?!\$)(?P<dollar_inline>.+?)(?<!\\)\$|"
    r"\\begin\s*\{(?P<math_env>(?:equation|alignat|align|alignedat|aligned|gathered|gather|multline|eqnarray|flalign|displaymath|math)\*?)\}"
    r"(?P<env_content>.+?)\\end\s*\{(?P=math_env)\}", re.S | re.I)
_MATH_CONTENT_NAMES = ("dollar_display", "bracket_display", "paren_inline", "dollar_inline", "env_content")
_ROW_BREAK = re.compile(r"(?<!\\)\\\\(?:\s*\[[^\]]*\])?")
_RELATION = re.compile(r"\\(?:not|neq|ne|leqslant|geqslant|leq|geq|le|ge|lt|gt)(?![A-Za-z])|<=|>=|!=|[≤≥≠<>=]")
# A digit can immediately follow a TeX command (\\frac2r, \\geq6).
# These are numeric text tokens, not a certified constant/variable distinction.
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_NEGATION = re.compile(r"\b(?:not|never|no)\b|\\(?:not|neg|lnot|neq|ne)(?![A-Za-z])|[¬≠]|不存在|不成立|不能|不必|并非", re.I)
_UNIVERSAL = re.compile(r"\b(?:any|arbitrary|every|all)\b|\\forall(?![A-Za-z])|∀|任意|每个|所有", re.I)
_EXISTENTIAL = re.compile(r"\b(?:some|exists?)\b|\\exists(?![A-Za-z])|∃|存在", re.I)
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

_CORRECTION = re.compile(
    r"\berrat(?:um|a)\b|\bcorrigend(?:um|a)\b|\bwithdrawn\b|\bincorrect\b|"
    r"\bcounter[-\s]?examples?\b|\bcorrection\b|\bcorrected\s+(?:version|statement|theorem|proof)\b|"
    r"\b(?:error|mistake)\s+in\b|\bdoes\s+not\s+hold\b|\bfalse\s+(?:statement|claim|theorem)\b|"
    r"\bshould\s+(?:instead\s+)?read\b|勘误|更正|反例|撤回|有误|不成立", re.I)
_CHANGE = re.compile(
    r"\brevis(?:ed|ion)\b|\breplac(?:ed|es)\b|\bweaken(?:ed|ing)\b|\bstronger\b|\bweaker\b|"
    r"\badditional\s+(?:hypothes[ei]s|assumptions?|conditions?)\b|\bnow\s+assume\b|修订|改为|新增假设|弱化", re.I)
_RESTRICTED_USE = re.compile(
    r"\b(?:special|particular)\s+case\b|\bwe\s+only\s+need\b|\bunder\s+(?:the\s+)?additional\s+assumptions?\b|"
    r"\brestrict(?:ed|ing)\s+to\b|\bwe\s+(?:use|apply)\b|这里只需|特例|仅使用|应用于", re.I)
_CITED_USE = re.compile(
    r"\bby\s+(?:the\s+)?(?:theorem|lemma|proposition|corollary)\b|\bapplying\b|"
    r"\b(?:theorem|lemma|proposition|corollary)\b.{0,70}?\\cite|由.{0,12}?(?:定理|引理|命题)", re.I | re.S)
_CITATION = re.compile(r"\\(?:cite\w*|autocite|parencite|textcite)\*?(?:\s*\[[^\]]*\])*\s*\{([^{}]*)\}", re.I)
_STYLE = re.compile(r"\\(?:left|right|[bB]igg?[lrm]?|displaystyle|textstyle|quad|qquad|enspace|thinspace|notag|nonumber|mathstrut|strut)(?![A-Za-z])|\\[,;!:]")
_FONT_WRAPPER = re.compile(r"\\(?:bm|boldsymbol|mathbf|mathit|mathrm|mathsf|mathtt|mathnormal)\s*\{([^{}]*)\}")
_ENV = re.compile(r"\\(?:begin|end)\s*\{[^{}]*\}")
_STOP_ANCHORS = frozenset("let suppose assume assuming then for any all every some there exist exists have has the a an and or but by with without that this these those thus hence where which whose such be been is are was were will shall if only into from of to in on at as we it its their they our result theorem lemma proposition corollary proof follows holds show proves each respectively".split())
_EQUIV_INTRO = re.compile(r"\b(?:the\s+)?following\s+(?:(?:statements?|conditions?|assertions?)\s+)?are\s+equivalent\b", re.I)
_INLINE_ITEM = re.compile(r"(?:\$\s*)?\(\s*(?:\{\s*\\rm\s+|\{\s*\\mathrm\s*\{?|\\mathrm\s*\{)?([ivxlcdm]+|\d+)\s*\}*\s*\)(?:\s*\$)?", re.I)
_OBJECT_PATTERN = r"\$[^$\r\n]{1,160}\$|\\\([^\r\n]{1,160}?\\\)|[A-Za-z][A-Za-z0-9_'/\\-]{0,30}"
_LET_DECL = re.compile(r"\b(?:let|suppose|assume)\s+(?P<object>" + _OBJECT_PATTERN + r")\s+(?:be|is)\s+(?P<description>[^.;\r\n]{1,220})", re.I)
_FOR_DECL = re.compile(r"\bfor\s+(?:any|every|each|all)\s+(?P<description>[^$.;\r\n]{1,140}?)\s*(?P<object>\$[^$\r\n]{1,160}\$)", re.I)
_BASE_TYPES = re.compile(r"\b(?:integral\s+domain|domain|field\s+extension|ideal\s+sheaf|representation|ring|group|space|solution|vector|element|module|field|morphism|map|function|operator|ideal|functor|process|distribution|variety|scheme|subgroup|point|line|manifold|curve|surface)\b", re.I)
_DOC_CHANGE = re.compile(r"\berrat(?:um|a)\b|\bcorrigend(?:um|a)\b|\b(?:mistake|incorrect|withdrawn|correction|corrected|replaced|revised|weakened|weaker|adjusted)\b|\b(?:missing|omitted|additional)\s+(?:assumption|hypothesis|condition)\b|\berror\s+in\b|勘误|更正|补充假设|漏掉.{0,8}假设|撤回", re.I)
_DOC_REVISION = re.compile(r"\berrat(?:um|a)\b|\bcorrigend(?:um|a)\b|\b(?:mistake|incorrect|withdrawn|corrected|correction|replaced|revised|weakened)\b|\b(?:missing|omitted|additional)\s+(?:assumption|hypothesis|condition)\b|\berror\s+in\b|勘误|更正|补充假设|漏掉.{0,8}假设|撤回", re.I)


def _string(value) -> str:
    return value if isinstance(value, str) else "" if value is None else str(value)


def _mask(match) -> str:
    """Blank inactive characters while preserving source character coordinates."""
    return re.sub(r"[^\r\n]", " ", match.group(0))


def _active_text(text: str) -> str:
    text = re.sub(r"(?<!\\)%[^\r\n]*", _mask, text)
    text = re.sub(r"\\begin\s*\{comment\}.*?\\end\s*\{comment\}", _mask, text, flags=re.S | re.I)
    return text


def _semantic_active(text: str) -> str:
    text = _active_text(text)
    # Unescaped & is alignment syntax inside a math block. Escaped \\& is
    # retained. Preserve all coordinates rather than rebuilding TeX strings.
    chars = list(text)
    for block in _MATH_BLOCK.finditer(text):
        for match in re.finditer(r"(?<!\\)&", block.group(0)):
            chars[block.start() + match.start()] = " "
        for match in _ROW_BREAK.finditer(block.group(0)):
            for position in range(block.start() + match.start(), block.start() + match.end()):
                if chars[position] not in "\r\n":
                    chars[position] = " "
    text = "".join(chars)
    text = _META.sub(_mask, text)
    text = _STYLE.sub(_mask, text)
    text = _ENV.sub(_mask, text)
    return text


def _clean(text: str) -> str:
    text = _semantic_active(text)
    return unicodedata.normalize("NFKC", text).casefold()


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(_clean(text))[:MAX_TOKENS]


def _normalized(text: str) -> str:
    return " ".join(_tokens(text))


def _similarity(old: str, new: str) -> float:
    a, b = Counter(_tokens(old)), Counter(_tokens(new))
    if not a or not b:
        return 0.0
    return round(sum((a & b).values()) / sum((a | b).values()), 6)


def _quote(text: str, start: int = 0, end: int | None = None, radius: int = 100) -> str:
    end = min(len(text), start + 35) if end is None else end
    left, right = max(0, start - radius), min(len(text), end + radius)
    value = text[left:right].strip()
    return ("…" if left else "") + value + ("…" if right < len(text) else "")


def _paper_id(side: dict) -> str:
    return re.sub(r"v\d+$", "", _string(side.get("paper_id")).strip(), flags=re.I)


def _version(side: dict) -> int | None:
    value = side.get("version")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    match = re.fullmatch(r"v?([1-9]\d*)", _string(value).strip(), re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"v([1-9]\d*)$", _string(side.get("paper_id")), re.I)
    return int(match.group(1)) if match else None


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


def _scope(text: str, position: int) -> str:
    # This is a syntactic hint, never a logical premise/conclusion certificate.
    items = list(re.finditer(r"\\item\b(?:\[[^\]]*\])?", text[:position], re.I))
    segment_start = items[-1].end() if items else 0
    segment = text[segment_start:]
    local_position = position - segment_start
    boundary = re.search(r"\bthen\b|\bwe\s+have\b|\bit\s+follows\b|\\(?:Longrightarrow|implies)(?![A-Za-z])|⇒", segment, re.I)
    if boundary:
        return "premise_like_text" if local_position < boundary.start() else "conclusion_like_text"
    prefix = text[max(segment_start, position - 180):position]
    if re.search(r"\b(?:let|suppose|assume|assuming|if|provided|for)\b|假设|设|若", prefix, re.I):
        return "premise_like_text"
    return "statement_position_uncertain"


def _signal(kind, label, old_quote, new_quote, explanation, scope="statement_text", **extra):
    return dict(kind=kind, label=label, old_quote=old_quote, new_quote=new_quote,
                explanation=explanation, scope=scope, **extra)


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


def _math_normalize(text: str) -> str:
    text = _active_text(text)
    text = _META.sub("", text)
    text = _ENV.sub("", text)
    text = re.sub(r"\\(?:notag|nonumber)(?![A-Za-z])", "", text)
    text = re.sub(r"(?<!\\)&", "", text)
    text = _ROW_BREAK.sub("", text)
    text = _STYLE.sub("", text)
    text = text.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    # Sentence punctuation is often placed just inside a math delimiter. Do not
    # strip ! (factorial), inequality symbols, negation, or tuple-internal commas.
    return re.sub(r"\s+", "", text).rstrip(".,;")


def _notation_projection(text: str) -> str:
    """A display comparison only: bold/roman notation can carry meaning."""
    for _ in range(4):
        changed = _FONT_WRAPPER.sub(lambda m: m.group(1), text)
        if changed == text:
            break
        text = changed
    return text


def _formula_evidence(text: str, unit: dict) -> dict:
    """Keep inline formulas with their complete local conditional sentence.

    Math is masked only for finding prose sentence boundaries. Returned quotes
    remain substrings of the active statement, with distinct formula coordinates.
    Multiline rows retain their exact row quote and a separate prose context.
    """
    active = _active_text(text)
    prose = list(active)
    for block in _MATH_BLOCK.finditer(active):
        for position in range(block.start(), block.end()):
            if prose[position] not in "\r\n":
                prose[position] = " "
    prose = "".join(prose)
    start, end = unit["start"], unit["end"]
    left, right = 0, len(active)
    for match in re.finditer(r"[.!?](?=\s|$)|\\item(?:\[[^\]]*\])?", prose):
        if re.search(r"(?:e\.g|i\.e|cf|eq|thm|prop|lem|fig)$", prose[max(0, match.start() - 7):match.start()], re.I):
            continue
        if match.end() <= start:
            left = match.end()
        elif match.start() >= end:
            right = match.end()
            break
    while left < right and active[left].isspace():
        left += 1
    while right > left and active[right - 1].isspace():
        right -= 1
    # A displayed list can end with punctuation inside math. If its following
    # conditional belongs to the same prose sentence, start an inline trigger's
    # context at its own If rather than at the preceding unrelated list.
    conditional_starts = list(re.finditer(r"\bif\b", prose[left:start], re.I))
    if conditional_starts:
        candidate_left = left + conditional_starts[-1].start()
        if re.search(r"\bthen\b", prose[candidate_left:right], re.I):
            left = candidate_left
    truncated = right - left > 2200
    if truncated:
        left, right = max(left, start - 850), min(right, max(end, start + 1350))
    inline = unit["environment"] in {"dollar_inline", "paren_inline"}
    quote_start, quote_end = (left, right) if inline else (start, end)
    return {"quote": active[quote_start:quote_end], "span": [quote_start, quote_end],
            "formula_quote": active[start:end], "formula_span": [start, end],
            "context_quote": active[left:right], "context_span": [left, right],
            "context_truncated": truncated,
            "context_scope": "bounded_same_prose_sentence_not_complete_logical_scope"}


def _formula_backreference(source_text: str, other_text: str, unit: dict) -> dict | None:
    labels = re.findall(r"\\label\s*\{([^{}]+)\}", source_text[unit["start"]:unit["end"]])
    if not labels:
        return None
    for label in labels:
        match = re.search(r"\\(?:eqref|[cCvV]?[rR]ef|[cC]ref)\s*\{" + re.escape(label) + r"\}", other_text)
        if match:
            evidence = _formula_evidence(other_text, {"start": match.start(), "end": match.end(), "environment": "dollar_inline"})
            return {"label": label, "quote": evidence["quote"], "span": evidence["span"],
                    "reference_span": [match.start(), match.end()],
                    "status": "same_label_text_backreference_only_not_mathematical_equivalence"}
    return None


def _math_units(text: str) -> list[dict]:
    """Extract bounded active math rows, keeping branch conditions separate.

    This is TeX-aware text extraction, not a mathematical expression parser.
    An inherited comparison object in a cases environment is explicitly marked.
    """
    text = _active_text(text)
    values = []
    for block in _MATH_BLOCK.finditer(text):
        name = next(name for name in _MATH_CONTENT_NAMES if block.group(name) is not None)
        content = block.group(name)
        content_start = block.start(name)
        environment = block.group("math_env") or name
        has_cases = bool(re.search(r"\\begin\s*\{(?:cases|dcases|rcases)\*?\}", content, re.I))
        row_ranges, previous = [], 0
        for separator in _ROW_BREAK.finditer(content):
            row_ranges.append((previous, separator.start()))
            previous = separator.end()
        row_ranges.append((previous, len(content)))
        inherited_object, inherited_relation = "", ""
        for row_index, (start, end) in enumerate(row_ranges[:40]):
            raw = content[start:end]
            if not raw.strip() or len(raw) > 3000:
                continue
            core, condition = raw, ""
            # TeX prose conditions in align rows, e.g. &\\text{if $n>3$}.
            condition_match = re.search(r"\\(?:text|textrm|mbox)\s*\{\s*(?:if|for|when|whenever|otherwise)\b", core, re.I)
            if condition_match:
                condition = core[condition_match.start():].strip()
                core = core[:condition_match.start()].rstrip(" &\r\n\t")
            elif has_cases:
                amp = re.search(r"(?<!\\)&", core)
                if amp:
                    condition = core[amp.end():].strip()
                    core = core[:amp.start()]
            normalized = _math_normalize(core)
            relation = _RELATION.search(normalized)
            object_key, relation_key, inherited = "", "", False
            if relation:
                candidate_object = normalized[:relation.start()]
                if re.search(r"[A-Za-z]|[\u3400-\u9fff]", candidate_object):
                    object_key, relation_key = candidate_object, relation.group(0)
                    inherited_object, inherited_relation = object_key, relation_key
            elif has_cases and inherited_object:
                object_key, relation_key, inherited = inherited_object, inherited_relation, True
                normalized = object_key + relation_key + normalized
            if not normalized:
                continue
            if (values and values[-1]["block_start"] == block.start() and not has_cases
                    and not object_key and not condition
                    and re.match(r"^(?:[+\-]|\\(?:cdot|times|cap|cup)(?![A-Za-z]))", normalized)):
                # A display split over two typographical lines, e.g. A=B\\&+C,
                # is one expression rather than a new mathematical branch.
                values[-1]["normalized"] += normalized
                values[-1]["end"] = content_start + end
                continue
            # One-row quotes include delimiters; multirow quotes show the exact
            # row including its branch condition, not a later proof or $condition$.
            quote_start = block.start() if len(row_ranges) == 1 else content_start + start
            quote_end = block.end() if len(row_ranges) == 1 else content_start + end
            values.append({
                "normalized": normalized, "start": quote_start, "end": quote_end,
                "raw_core": core,
                "object": object_key, "relation": relation_key,
                "condition": condition, "environment": environment,
                "row_index": row_index, "block_start": block.start(),
                "row_count": len(row_ranges), "inherited_object": inherited,
            })
            if len(values) >= 80:
                return values
        if len(values) >= 80:
            break
    return values


def _formulas(text: str) -> list[tuple[str, int, int]]:
    return [(u["normalized"], u["start"], u["end"]) for u in _math_units(text)]


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


def _title(text: str) -> str:
    active = _active_text(text)
    begin = _BEGIN_DECL.search(active)
    if not begin:
        return ""
    rest = active[begin.end():].lstrip()
    if not rest.startswith("["):
        return ""
    depth = 0
    for i, char in enumerate(rest[:700]):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return re.sub(r"\s+", " ", _META.sub("", rest[1:i])).strip()[:500]
    return ""


def _anchor_words(text: str) -> set[str]:
    return {t for t in _tokens(text) if len(t) >= 3 and t[0].isalpha() and t not in _STOP_ANCHORS and not t.isdigit()}


def _conclusion(text: str) -> str:
    active = _active_text(text)
    match = re.search(r"\bthen\b|\bwe\s+have\b|\\(?:Longrightarrow|implies)(?![A-Za-z])|⇒", active, re.I)
    return active[match.end():match.end() + 650].strip() if match else ""


def _math_profile(text: str) -> dict:
    active = _active_text(text)
    marker = re.search(r"\bthen\b|\bwe\s+have\b|\\(?:Longrightarrow|implies)(?![A-Za-z])|⇒", active, re.I)
    units = _math_units(active)
    if marker:
        units = [u for u in units if u["start"] >= marker.end()]
    counts = Counter(u["object"] for u in units if u["object"])
    return {"comparison_objects": sorted(counts), "object_occurrences": dict(counts),
            "formula_rows": len(units), "has_branch_structure": any(u["row_count"] > 1 for u in units),
            "premise_text": active[:marker.start()] if marker else ""}


def _alignment(old: str, new: str, similarity: float) -> dict:
    at, bt = _title(old), _title(new)
    aw, bw = _anchor_words(at), _anchor_words(bt)
    title_mismatch = bool(aw and bw and aw != bw)
    anchors = sorted(_anchor_words(old) & _anchor_words(new))[:15]
    af, bf = _formulas(old), _formulas(new)
    # A shared $x$ is not enough to align two mathematical statements.
    afs = {v for v, _, _ in af if len(_TOKEN.findall(v)) >= 4 or _RELATION.search(v)}
    bfs = {v for v, _, _ in bf if len(_TOKEN.findall(v)) >= 4 or _RELATION.search(v)}
    common_formulas = sorted(afs & bfs)
    ac, bc = _conclusion(old), _conclusion(new)
    conclusion_similarity = _similarity(ac, bc) if ac and bc else None
    conclusion_mismatch = bool(ac and bc and conclusion_similarity < .20)
    ap, bp = _math_profile(old), _math_profile(new)
    shared_objects = sorted(set(ap["comparison_objects"]) & set(bp["comparison_objects"]))
    object_mismatch = bool(ap["comparison_objects"] and bp["comparison_objects"] and not shared_objects)
    av, bv = Counter(_tokens(old)), Counter(_tokens(new))
    common_tokens = sum((av & bv).values())
    old_coverage = common_tokens / max(1, sum(av.values()))
    new_coverage = common_tokens / max(1, sum(bv.values()))
    premise_similarity = _similarity(ap["premise_text"], bp["premise_text"]) if ap["premise_text"] and bp["premise_text"] else None
    repeated_object = any(
        (ap["object_occurrences"].get(obj, 0) == 1 and bp["object_occurrences"].get(obj, 0) >= 2) or
        (bp["object_occurrences"].get(obj, 0) == 1 and ap["object_occurrences"].get(obj, 0) >= 2)
        for obj in shared_objects)
    branch_expansion = bool(
        repeated_object and (ap["has_branch_structure"] or bp["has_branch_structure"])
        and premise_similarity is not None and premise_similarity >= .85
        and max(old_coverage, new_coverage) >= .90 and len(anchors) >= 4
        and not title_mismatch and not object_mismatch)
    same = _normalized(old) == _normalized(new) and bool(_normalized(old))
    # Short statements rarely have four content anchors; accept them when both
    # sides are mostly covered and the located conclusions nearly coincide.
    short_match = (min(old_coverage, new_coverage) >= .75 and len(anchors) >= 2
                   and conclusion_similarity is not None and conclusion_similarity >= .9)
    strong = similarity >= .65 and (bool(common_formulas) or len(anchors) >= 4 or short_match) and not conclusion_mismatch
    if same:
        status, label = "same_extracted_text", "有界活跃声明词元一致；全局上下文等价仍未知"
    elif title_mismatch:
        status, label = "named_statement_mismatch", "结果标题不同，需先对齐命题，不能先按条件变化高排"
    elif object_mismatch:
        status, label = "comparison_object_mismatch", "结论比较对象不同，先核对是否同一命题或记号替换"
    elif conclusion_mismatch:
        status, label = "conclusion_alignment_uncertain", "结论位置的文字对齐较弱；可能比较了不同结果"
    elif strong:
        status, label = "strong_textual_overlap", "共同正文与公式锚点支持候选对齐；尚非数学同一性认证"
    elif similarity >= .30:
        status, label = "partial_textual_overlap", "只有部分文字对齐，命题对应关系待核"
    else:
        status, label = "weak_textual_overlap", "共同锚点不足，优先核对是否同一命题"
    match_score = similarity * .8 + (min(2, len(common_formulas)) * .05) + (.1 if at and at == bt else 0)
    match_score -= .20 if title_mismatch else 0
    match_score -= .10 if conclusion_mismatch else 0
    match_score -= .15 if object_mismatch else 0
    if branch_expansion:
        match_score += .10 * max(old_coverage, new_coverage) + .10 * premise_similarity
    return {"status": status, "label": label, "old_title": at, "new_title": bt,
            "common_anchors": anchors, "common_formula_count": len(common_formulas),
            "common_formula_examples": common_formulas[:2],
            "old_conclusion_excerpt": ac, "new_conclusion_excerpt": bc,
            "conclusion_location_basis": "first_syntactic_then_or_implication_only_not_logical_parse",
            "conclusion_text_similarity": conclusion_similarity,
            "title_mismatch": title_mismatch, "conclusion_mismatch": conclusion_mismatch,
            "comparison_object_mismatch": object_mismatch,
            "old_comparison_objects": ap["comparison_objects"], "new_comparison_objects": bp["comparison_objects"],
            "shared_comparison_objects": shared_objects,
            "old_token_coverage_by_new": round(old_coverage, 6), "new_token_coverage_by_old": round(new_coverage, 6),
            "premise_text_similarity": premise_similarity,
            "branch_expansion_candidate": branch_expansion,
            "branch_expansion_basis": "共同前提、共同比较对象的一对多分支、单侧高覆盖；还需本批竞争候选门控。" if branch_expansion else None,
            "candidate_match_score": round(max(0, min(1, match_score)), 6),
            "same_mathematical_statement_certified": False}


def _add_competition(records: list[dict]) -> None:
    # Only compare alternatives for the SAME source statement and SAME target
    # paper/version. A later paper is never suppressed merely because some other
    # paper has a closer candidate; broad adoption observations stay possible.
    forward, backward = defaultdict(dict), defaultdict(dict)
    forward_refs, backward_refs = defaultdict(dict), defaultdict(dict)
    keys = []
    for i, r in enumerate(records):
        a, b = r["old"], r["new"]
        ak = (_paper_id(a), _version(a), sha256(_normalized(a["text"]).encode("utf-8")).hexdigest()[:24])
        bk = (_paper_id(b), _version(b), sha256(_normalized(b["text"]).encode("utf-8")).hexdigest()[:24])
        fk, rk = (ak, bk[:2]), (bk, ak[:2])
        score = r["algorithm"]["alignment_evidence"]["candidate_match_score"]
        forward[fk][bk] = max(score, forward[fk].get(bk, 0))
        backward[rk][ak] = max(score, backward[rk].get(ak, 0))
        forward_refs[fk][bk] = {"statement_id": b.get("statement_id"), "pair_id": r["pair_id"],
                                "paper_id": b.get("paper_id"), "version": b.get("version"), "match_score": score}
        backward_refs[rk][ak] = {"statement_id": a.get("statement_id"), "pair_id": r["pair_id"],
                                 "paper_id": a.get("paper_id"), "version": a.get("version"), "match_score": score}
        keys.append((fk, rk, ak, bk))

    def statistics(groups):
        output = {}
        for group_key, group in groups.items():
            top = max(group.values())
            count = sum(value == top for value in group.values())
            second = max((value for value in group.values() if value < top), default=None)
            best_keys = sorted(group, key=lambda key: (-group[key], repr(key)))[:2]
            output[group_key] = (len(group), top, count, second, best_keys)
        return output

    fs, bs = statistics(forward), statistics(backward)

    def summary(group, key, stats, refs):
        own = group[key]
        size, top, count, second, best_keys = stats
        alternative = top if own < top or count > 1 else second
        alternative_key = next((candidate for candidate in best_keys if candidate != key), None)
        return {"distinct_statement_candidates": size, "pair_is_best": alternative is None or own >= alternative,
                "match_score": own, "best_alternative_score": alternative,
                "best_alternative": refs.get(alternative_key),
                "margin_over_best_alternative": round(own - alternative, 6) if alternative is not None else None}

    for r, (fk, rk, ak, bk) in zip(records, keys):
        f, b = summary(forward[fk], bk, fs[fk], forward_refs[fk]), summary(backward[rk], ak, bs[rk], backward_refs[rk])
        margins = [x["margin_over_best_alternative"] for x in (f, b) if x["margin_over_best_alternative"] is not None]
        ambiguous = bool(margins and min(margins) <= .035)
        not_best = not (f["pair_is_best"] and b["pair_is_best"])
        alignment = r["algorithm"]["alignment_evidence"]
        local_competition = {"forward": f, "reverse": b,
            "reciprocal_best_in_input": not not_best, "near_tie_or_better_alternative": ambiguous,
            "scope": "current_input_alternatives_within_same_target_paper_and_version_only"}
        cached = r.get("cached_alignment_competition")
        cached_competition = cached.get("competition") if isinstance(cached, dict) else None
        if isinstance(cached_competition, dict):
            # A selected recheck batch omits alternatives from the original
            # retrieval frame. Its apparent uniqueness cannot remove an older
            # automatic not-best or near-tie restriction.
            cached_sides = [cached_competition.get(k, {}) for k in ("forward", "reverse")]
            not_best |= cached_competition.get("reciprocal_best_in_input") is False or any(v.get("pair_is_best") is False for v in cached_sides)
            cached_margins = [v.get("margin_over_best_alternative") for v in cached_sides]
            cached_margins = [v for v in cached_margins if isinstance(v, (int, float))]
            margins.extend(cached_margins)
            ambiguous |= cached_competition.get("near_tie_or_better_alternative") is True or bool(cached_margins and min(cached_margins) <= .035)
            alignment["competition"] = dict(cached_competition)
            alignment["competition"]["scope"] = "provided_prior_full_frame_with_current_input_restrictions_retained"
            alignment["competition"]["reciprocal_best_in_input"] = not not_best
            alignment["competition"]["near_tie_or_better_alternative"] = ambiguous
            alignment["competition_provenance"] = {k: v for k, v in cached.items() if k != "competition"}
            alignment["competition_provenance"].update(
                interpretation="supplied_prior_automatic_frame_not_loaded_from_files_not_human_labels",
                external_hashes_reverified_by_engine=False, subset_uniqueness_cannot_upgrade=True)
            alignment["current_input_competition"] = local_competition
        else:
            alignment["competition"] = local_competition
            alignment["competition_provenance"] = {"scope": "current_input_only",
                "subset_uniqueness_is_global_uniqueness": False,
                "note": "本批互选只覆盖所给输入，未证明原检索框内没有其他竞争声明。"}
        if (alignment["branch_expansion_candidate"] and not not_best and not ambiguous
                and (not margins or min(margins) >= .10)):
            alignment.update(status="branch_expansion_textual_match", label="共同前提与比较对象支持单式/分支扩展对齐；数学强度仍未知")
        elif alignment["status"] not in {"same_extracted_text", "named_statement_mismatch", "comparison_object_mismatch", "conclusion_alignment_uncertain"}:
            if not_best:
                alignment.update(status="not_best_statement_candidate", label="同一目标文档内存在更近的竞争声明，先核对命题对齐")
            elif ambiguous:
                alignment.update(status="competing_candidates_ambiguous", label="同一目标文档存在相近竞争声明，匹配尚不唯一")
        if isinstance(cached, dict):
            prior_status = cached.get("statement_alignment")
            blocked = {"named_statement_mismatch", "comparison_object_mismatch", "conclusion_alignment_uncertain",
                       "not_best_statement_candidate", "competing_candidates_ambiguous", "partial_textual_overlap", "weak_textual_overlap"}
            if prior_status in blocked and alignment["status"] in {"same_extracted_text", "strong_textual_overlap", "branch_expansion_textual_match"}:
                alignment.update(status=prior_status, label="沿用所给完整旧候选框中的对齐阻断/不确定性；子集无竞争不构成升级证据")
        r["algorithm"]["statement_alignment"] = alignment["status"]


def _disclosure(old: dict, new: dict, oldtext: str, newtext: str) -> dict:
    matches = []
    general = []
    windows = [("old", "statement", oldtext), ("new", "statement", newtext),
               ("old", "context", _string(old.get("context"))[:MAX_CONTEXT]),
               ("new", "context", _string(new.get("context"))[:MAX_CONTEXT])]
    for side, scope, window in windows:
        window = _active_text(window)
        for pattern, collector in ((_CORRECTION, matches), (_CHANGE, general)):
            for match in pattern.finditer(window):
                if len(collector) >= 8:
                    break
                # An explicit denial of an error is not an error admission.
                prefix = window[max(0, match.start() - 25):match.start()].casefold()
                if re.search(r"\b(?:no|not|without)\s+(?:an?\s+)?$", prefix):
                    continue
                collector.append({"side": side, "scope": scope, "term": match.group(0), "quote": _quote(window, match.start(), match.end())})
    if matches:
        status, label = "matched_correction_language", "给定窗口命中更正或纠错措辞（与本变化的关联待核）"
    elif general:
        status, label = "matched_change_language", "给定窗口命中一般修订措辞（尚未确认解释本变化）"
    elif any(window.strip() for _, _, window in windows):
        status, label = "not_matched_in_provided_window", "给定窗口未命中说明词；不能据此认证悄然修改"
    else:
        status, label = "insufficient_window", "没有足够文本检查说明措辞"
    return {
        "status": status, "label": label, "matched_quotes": matches or general,
        "silent_change_certified": False,
        "coverage": {"scope": "provided_statement_and_bounded_local_context_only",
                     "context_char_limit_each_side": MAX_CONTEXT, "full_document_checked": False,
                     "linked_errata_or_version_comments_checked": False,
                     "old_context_present": bool(old.get("context")), "new_context_present": bool(new.get("context"))},
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
    active = _active_text(text)
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


_HYPOTHESIS = re.compile(r"(?<![A-Za-z])(?P<cue>if|suppose(?:\s+that)?|assume(?:\s+that)?|assuming(?:\s+that)?|provided(?:\s+that)?)\s+(?P<body>[^.;,\r\n]{1,220}?)(?=\s*(?:,|\.|;|\bthen\b|$))", re.I)


def _premise_region(text: str) -> str:
    marker = re.search(r"\bthen\b|\bwe\s+have\b|\\(?:Longrightarrow|implies)(?![A-Za-z])|⇒", text, re.I)
    return text[:marker.start()] if marker else ""


def _hypothesis_clauses(text: str) -> list[re.Match]:
    region = _premise_region(text)
    clauses = []
    for match in _HYPOTHESIS.finditer(region):
        if re.search(r"\bonly\s*$", region[:match.start()], re.I) or re.match(r"and\s+only\s+if\b", match.group("body"), re.I):
            continue
        clauses.append(match)
    return clauses


def _object_mention(text: str, key: str) -> tuple[int, int] | None:
    for unit in _math_units(text):
        if unit["normalized"] == key:
            return unit["start"], unit["end"]
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_']*", key):
        match = re.search(r"(?<![A-Za-z0-9_\\])" + re.escape(key) + r"(?![A-Za-z0-9_'])", _active_text(text))
        return (match.start(), match.end()) if match else None
    return None


def _new_premises(old: str, new: str) -> list[dict]:
    a, b = _active_text(old), _active_text(new)
    old_decls, new_decls = _declarations(a), _declarations(b)
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
        if not new_np:
            continue
        for od in old_decls:
            old_np = _noun_phrase(od["description"]) if od["object"] == nd["object"] else None
            if not old_np or old_np[0] != new_np[0]:
                continue
            old_words = {w.casefold() for w in old_np[1]}
            for word in new_np[1]:
                if word.casefold() in old_words:
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
    # A clause whose content words the old premise already carries (e.g. moved
    # from "Let $G$ be a planar graph") is not reported as new.
    old_premise = _normalized(_premise_region(a))
    old_premise_words = _anchor_words(_premise_region(a))
    old_bodies = {_normalized(m.group("body")) for m in _hypothesis_clauses(a)}
    for match in _hypothesis_clauses(b):
        body = _normalized(match.group("body"))
        if not body or body in old_bodies or body in old_premise:
            continue
        if any(d["start"] < match.end() and match.start() < d["end"] for d in new_decls):
            continue  # "Suppose $G$ is a planar graph" is compared as a declaration above.
        # Formula-only clauses ("If $n\\geq 3$") are left to formula_scope.
        words = _anchor_words(re.sub(r"\$[^$]*\$", " ", match.group("body")))
        if not words or words <= old_premise_words:
            continue
        units = _math_units(match.group("body"))
        obj = units[0]["normalized"] if units else ""
        om = _object_mention(a, obj) if obj else None
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


_REFERENCE_NUMBER = re.compile(r"\b(theorem|thm|lemma|lem|proposition|prop|corollary|cor|claim|definition|defn)\.?[\s~{}]*(\d+(?:\.\d+)*[a-z]?)\b", re.I)
_REFERENCE_KIND = {"thm": "theorem", "lem": "lemma", "prop": "proposition", "cor": "corollary", "defn": "definition"}


def _reference_mentions(text: str, include_foreign: bool = False) -> dict[str, str]:
    values = {}
    citations = list(_CITATION.finditer(text))
    for match in _REFERENCE_NUMBER.finditer(text):
        if not include_foreign:
            in_citation_argument = any(c.start() <= match.start() < c.end() for c in citations)
            external_owner = re.match(r"\s+(?:of|by|in)\s+(?:the\s+)?\\(?:cite\w*|autocite|parencite|textcite)\b", text[match.end():], re.I)
            if in_citation_argument or external_owner:
                continue
        kind = _REFERENCE_KIND.get(match.group(1).casefold(), match.group(1).casefold())
        values[kind + ":" + match.group(2).casefold()] = match.group(0)
    for match in re.finditer(r"\\(?:[cCvV]?[cC]?ref|eqref)\*?\s*\{([^{}]*)\}", text):
        if not include_foreign and any(c.start() <= match.start() < c.end() for c in citations):
            continue
        for key in match.group(1).split(","):
            if key.strip():
                values["label:" + key.strip()] = match.group(0)
    return values


def _statement_references(side: dict, text: str) -> dict[str, str]:
    values = {}
    supplied = side.get("statement_reference_keys", [])
    if isinstance(supplied, list):
        for item in supplied:
            item = _string(item).strip()
            if not item:
                continue
            parsed = _reference_mentions(item)
            if parsed:
                values.update(parsed)
            elif re.fullmatch(r"[A-Za-z0-9_:/.-]{1,150}", item) and not item.isdigit():
                values["label:" + item.removeprefix("label:")] = item
    active = _active_text(text)
    declaration = _BEGIN_DECL.search(active)
    header = active[declaration.end():] if declaration else active
    header = re.sub(r"^\s*\[[^\]]{0,500}\]", "", header)
    # Body citations to other lemmas are not the current statement's number.
    leading = re.match(r"\s*((?:\\label\s*\{[^{}]*\}\s*)+)", header)
    if leading:
        for match in re.finditer(r"\\label\s*\{([^{}]*)\}", leading.group(0)):
            values["label:" + match.group(1)] = match.group(1)
    numbered_header = re.match(r"\s*(?:Theorem|Lemma|Proposition|Corollary|Claim|Definition)\.?\s+\d+(?:\.\d+)*[a-z]?\b", header, re.I)
    if numbered_header:
        values.update(_reference_mentions(numbered_header.group(0)))
    return values


def _document_scan(text: str, cache: dict, source_type: str = "unknown") -> dict:
    # sha256 in the supplied metadata may identify the containing HTML file;
    # comments and abstract from that HTML have different text. Cache by the
    # actual text digest, not by the supplied file hash alone.
    digest = sha256(text.encode("utf-8")).hexdigest()
    key = (source_type, digest, len(text))
    if key in cache:
        return cache[key]
    active = _active_text(text[:MAX_EVIDENCE_TEXT])
    active_end = re.search(r"\\end\s*\{document\}", active) if source_type == "full_tex" else None
    excluded_post_document = len(active) - active_end.end() if active_end else 0
    if active_end:
        active = active[:active_end.end()]
    matches, total = [], 0
    for match in _DOC_CHANGE.finditer(active):
        if re.search(r"\b(?:no|not|without)\s+(?:an?\s+)?$", active[max(0, match.start() - 30):match.start()], re.I):
            continue
        total += 1
        if len(matches) >= 256:
            continue
        # A paragraph/sentence window preserves the reference next to the cue.
        # Decimal theorem numbers do not create sentence boundaries.
        left_floor, right_ceiling = max(0, match.start() - 500), min(len(active), match.end() + 500)
        before = active[left_floor:match.start()]
        boundaries = list(re.finditer(r"\n\s*\n|[.!?](?=\s+[A-Z\\])", before))
        left = left_floor + boundaries[-1].end() if boundaries else left_floor
        after = active[match.end():right_ceiling]
        boundary = re.search(r"\n\s*\n|[.!?](?=\s+[A-Z\\]|\s*$)", after)
        right = match.end() + boundary.end() if boundary else right_ceiling
        while left < right and active[left].isspace():
            left += 1
        while right > left and active[right - 1].isspace():
            right -= 1
        matches.append({"term": match.group(0), "cue_span": [match.start(), match.end()],
                        "quote": active[left:right], "span": [left, right]})
    result = {"text_digest": digest, "input_characters": len(text), "scanned_characters": len(active),
              "truncated": len(text) > MAX_EVIDENCE_TEXT, "matches": matches,
              "cue_count": total, "cue_limit_reached": total > len(matches),
              "bounded_to_first_active_end_document": bool(active_end),
              "post_document_characters_excluded": excluded_post_document}
    if len(cache) >= MAX_EVIDENCE_CACHE:
        cache.pop(next(iter(cache)))
    cache[key] = result
    return result


def _proof_only_notice(quote: str, matched_refs: dict[str, str]) -> bool:
    for literal in matched_refs.values():
        position = quote.find(literal)
        if position < 0:
            continue
        if re.search(r"\bproof\s+(?:of\s+)?(?:the\s+)?$", quote[max(0, position - 35):position], re.I):
            return not bool(re.search(r"\b(?:missing|omitted|additional|new)\s+(?:assumption|hypothesis|condition)|\bassum(?:e|ption)\b", quote[position:position + 200], re.I))
    return False


def _full_tex_revision_notice(quote: str) -> bool:
    """Require revision syntax, not an error/correction word near a result.

    This projection is used only for qualifying a notice. Returned quotations
    and offsets still refer to the original, comment-masked supplied text.
    A bare additional assumption in the current theorem cannot disclose its
    own addition. Nor can a numerical error become a revision by proximity to
    a theorem number.
    """
    def citation_prose(match):
        options = re.findall(r"\[([^\]]*)\]", match.group(0))
        return " " + " ".join(options) + " " if options else " citedwork "

    prose = _CITATION.sub(citation_prose, quote)
    prose = re.sub(r"\s+", " ", prose)
    article = r"(?:(?:the|a|an|our|this|that|original|earlier|previous|published|revised)\s+){0,3}"
    identifier = r"(?:\d+(?:\.\d+)*[a-z]?(?![A-Za-z0-9]|\.\d)|\\(?:[cCvV]?ref|[cC]ref)\*?\s*\{[^{}]*\})"
    result = r"\b(?:theorem|thm|lemma|lem|proposition|prop|corollary|cor|claim|statement|conjecture|definition|proof|argument)\b(?:[\s~]*" + identifier + r")?"
    # Direct ownership complements remain typed results, not arbitrary prose
    # gaps such as 'the error estimate given by Theorem 1.1'.
    subject = article + result + r"(?:\s+(?:of|for|in)\s+" + article + result + r")?(?:\s+(?:of|in|from)\s+citedwork)?"
    version = (r"\b(?:previous|earlier|old|original|first|second|revised)\s+(?:version|draft)\b"
               r"(?:\s+of\s+" + article + r"(?:paper|article|preprint|manuscript)\b|(?!\s+of\b))")
    publication = r"(?:" + version + r"|\b(?:previous|earlier|old|original)\s+(?:paper|article|preprint|manuscript)\b)"
    change = r"(?:corrected|revised|amended|adjusted|replaced|weakened|withdrawn|retracted)"
    copula = r"\s+(?:is|are|was|were|(?:has|have|had)(?:\s+now)?\s+been)(?:\s+(?:now|here|later|subsequently|partially))?\s+"
    predicate = copula + r"(?:incorrect|false|wrong|erroneous|invalid|incomplete|" + change + r")\b"
    for match in re.finditer(r"(?:" + subject + r"|" + article + publication + r")" + predicate, prose, re.I):
        # In 'the model error in Theorem 1.1 is corrected', the subject being
        # corrected is the error, not the theorem. Do not promote a reference
        # inside a prepositional phrase to the grammatical claim subject.
        if not re.search(r"\b(?:in|of|for|by|from|using|under|within|to)\s*$", prose[:match.start()], re.I):
            return True
    active_change = r"\b(?:correct(?:ed|s)?|revis(?:e|ed|es)|amend(?:ed|s)?|withdraw(?:n|s)?|replac(?:e|ed|es)|weaken(?:ed|s)?|adjust(?:ed|s)?)\s+"
    object_end = r"(?=\s*(?:$|[.,;:!?)\]}]|\b(?:and|by|in|of|to|with|from|that|which|as|because|after|before|under|since|is|are|was|were|has|have)\b))"
    if re.search(active_change + r"(?:" + subject + r"|" + article + publication + r")" + object_end, prose, re.I):
        return True
    if re.search(r"\b(?:correction|revision|amendment|withdrawal|retraction)\s+(?:of|to)\s+" + subject + object_end, prose, re.I):
        return True
    # Errors in a proof/statement, or an explicit finding/repair of an error
    # in a named claim or publication, have different syntax from expanding,
    # bounding or reducing an estimation/model error.
    if re.search(r"\b(?:an?|the)\s+(?:error|mistake)\s+in\s+" + article + r"(?:proof|statement|argument)\b", prose, re.I):
        return True
    admission = r"\b(?:found|discovered|identified|point(?:ed|s|ing)?\s+out|correct(?:ed|s)?|fix(?:ed|es)?|rectif(?:y|ied))\s+(?:an?|the)\s+(?:error|mistake)\s+in\s+"
    if re.search(admission + r"(?:" + subject + r"|" + article + publication + r")" + object_end, prose, re.I):
        return True
    if re.search(r"\b(?:a|the)\s+mistake\s+in\s+" + subject + copula + change + r"\b", prose, re.I):
        return True
    # An explicit omission plus a restoration predicate can explain an added
    # premise without a printed theorem number. 'Additional assumption' alone
    # never takes this branch.
    missing = r"\b(?:missing|omitted|overlooked)\s+(?:assumptions?|hypothes(?:is|es)|conditions?|premises?)\b"
    restore = r"(?:correct(?:ed|s)?|fix(?:ed|es)?|repair(?:ed|s)?|restor(?:e|ed|es)|add(?:ed|s)?|includ(?:e|ed|es))"
    if re.search(r"\b" + restore + r"\s+" + article + missing, prose, re.I):
        return True
    if re.search(missing + r"[^.;]{0,180}" + copula + restore + r"\b", prose, re.I):
        return True
    if re.search(r"\b(?:an?|the)\s+(?:error|mistake)\s+in\s+" + article + publication + copula + change + r"\b", prose, re.I):
        return True
    # A publication/version comparison needs an actual premise-edit predicate
    # in the same clause, not a correction of a numerical quantity nearby.
    # A previous version of an algorithm is not a publication version.
    premise_edit = (r"\b(?:add(?:ed|s)?|includ(?:e|ed|es)|introduc(?:e|ed|es)|omit(?:ted|s)?|remov(?:e|ed|es)|drop(?:ped|s)?)\s+"
                    r"(?:(?!(?:to|for|of|in|with|under|that|when|which|is|was|are|were)\b)[A-Za-z][A-Za-z-]*\s+){0,7}"
                    r"(?:assumptions?|hypothes(?:is|es)|conditions?|premises?)\b")
    for clause in re.split(r";|(?<=[.!?])\s+(?=[A-Z])", prose):
        if (re.search(publication, clause, re.I)
                and re.search(premise_edit, clause, re.I)):
            return True
    # These are explicit notice genres, unlike bare error/correction terms.
    return bool(re.search(r"\berrat(?:um|a)\b|\bcorrigend(?:um|a)\b|勘误", prose, re.I))


def _revision_notice_candidate(quote: str, term: str, source_type: str, references: dict) -> bool:
    value = term.casefold()
    if source_type == "full_tex":
        return _full_tex_revision_notice(quote)
    # Mathematical error-correction, correction factors and substitutions are
    # not admissions that this paper's earlier statement was wrong.
    if value in {"correction", "corrected"} and re.search(
            r"\berror[-\s]+correct(?:ion|ed)\b|\bcorrection[-\s]+(?:codes?|properties|property|factors?|terms?|coefficients?|operators?)\b", quote, re.I):
        return False
    if re.search(r"errat|corrigend|mistake|incorrect|withdrawn|correct(?:ed|ion)|error\s+in|(?:missing|omitted|additional)\s+(?:assumption|hypothesis|condition)|勘误|更正|撤回", value, re.I):
        return True
    if source_type in {"version_comment", "public_erratum"}:
        return True  # Still needs a separate statement/attribute association.
    if re.search(r"\b(?:previous|earlier|old|revised|new|first|second)\s+versions?\b|\b(?:this|the)\s+revision\b|\bnow\s+(?:assume|correct|require)\b", quote, re.I):
        return True
    for literal in references.values():
        if re.search(re.escape(literal) + r"\s+(?:(?:has|have)\s+been|was|were|is\s+now|are\s+now)\s+(?:revised|adjusted|replaced|weakened|corrected|withdrawn)\b", quote, re.I):
            return True
    if (re.search(r"\b(?:present|this)\s+(?:note|paper)\b", quote, re.I)
            and re.search(r"\b(?:substitut\w*|replac\w*)\b", quote, re.I)
            and re.search(r"\b(?:claims?|statements?|theorems?)\b", quote, re.I)):
        return True
    return False


def _disclosure_documents(pair: dict, old: dict, new: dict, oldtext: str, newtext: str,
                          premises: list[dict], reverse: bool, cache: dict) -> tuple[dict, list[dict]]:
    """Scan only supplied text. document.version denotes the TARGET version.

    An independent erratum may use paper_id=target and version=None, retaining
    its own identity in source_paper_id/source_version. Its own v1 is never the
    corrected paper's v1. Paper association alone does not link a statement.
    """
    supplied = pair.get("evidence_documents", [])
    supplied = supplied if isinstance(supplied, list) else []
    refs = _statement_references(new, newtext)
    if _paper_id(old) == _paper_id(new):
        refs.update(_statement_references(old, oldtext))
    exported, findings, coverage_docs = [], [], []
    adequate, statement_notices, paper_notices = False, [], []
    for index, document in enumerate(supplied[:MAX_EVIDENCE_DOCUMENTS]):
        if not isinstance(document, dict):
            continue
        raw = _string(document.get("text"))
        available = document.get("source_available") is not False and bool(raw.strip())
        source_type = _string(document.get("source_type")) or "unknown"
        scan = _document_scan(raw, cache, source_type) if available else {"input_characters": len(raw), "scanned_characters": 0,
            "truncated": False, "matches": [], "cue_count": 0, "cue_limit_reached": False,
            "bounded_to_first_active_end_document": False, "post_document_characters_excluded": 0}
        doc_id = _string(document.get("doc_id")) or "doc_" + sha256((source_type + "\0" + _string(document.get("paper_id")) + "\0" + _string(document.get("version")) + "\0" + raw).encode("utf-8")).hexdigest()[:24]
        side = document.get("side")
        effective_side = {"old": "new", "new": "old"}.get(side, side) if reverse else side
        same_paper = bool(_paper_id(new)) and _paper_id(document) == _paper_id(new)
        nv, dv = _version(new), _version(document)
        old_version = _paper_id(old) == _paper_id(new) and dv is not None and dv == _version(old) and dv != nv
        historical = effective_side == "old" or old_version
        version_mismatch = nv is not None and dv is not None and nv != dv
        version_status = "matched_target_version" if nv is not None and nv == dv else "historical_or_other_version" if historical or version_mismatch else "one_or_both_versions_unknown"
        usable = available and same_paper and not historical and not version_mismatch
        associable = usable and (version_status == "matched_target_version" or source_type == "public_erratum")
        doc_adequate = usable and source_type in {"full_tex", "version_comment", "public_erratum", "abstract", "introduction"}
        adequate |= doc_adequate
        truncated = bool(scan["truncated"] or document.get("text_truncated"))
        metadata = {key: value for key, value in document.items() if key != "text"}
        metadata.update(doc_id=doc_id, effective_side=effective_side, text=raw[:400] if source_type != "full_tex" else "",
                        input_text_characters=len(raw), scanned_text_characters=scan["scanned_characters"],
                        text_excerpt_characters=min(400, len(raw)) if source_type != "full_tex" else 0,
                        text_excerpt_truncated=source_type != "full_tex" and len(raw) > 400,
                        scan_truncated=truncated, disclosure_scan_basis="supplied_text_only_no_path_or_url_opened",
                        target_version_relation=version_status)
        exported.append(metadata)
        coverage_docs.append({"doc_id": doc_id, "source_type": source_type, "paper_id": document.get("paper_id"),
            "version": document.get("version"), "side": side, "effective_side": effective_side,
            "source_available": available, "identity_basis": document.get("identity_basis"),
            "coverage_scope": document.get("coverage_scope"), "target_paper_matches": same_paper,
            "target_version_relation": version_status, "historical_background_only": historical,
            "eligible_for_target_notice_scan": usable, "eligible_for_statement_association": associable,
            "scanned_characters": scan["scanned_characters"], "input_characters": len(raw),
            "scan_truncated": truncated, "cue_count": scan["cue_count"], "cue_limit_reached": scan["cue_limit_reached"],
            "bounded_to_first_active_end_document": scan["bounded_to_first_active_end_document"],
            "post_document_characters_excluded": scan["post_document_characters_excluded"]})
        seen = set()
        for item in scan["matches"]:
            quote = item["quote"]
            mentions = _reference_mentions(quote)
            all_mentions = _reference_mentions(quote, include_foreign=True)
            notice_candidate = _revision_notice_candidate(quote, item["term"], source_type, all_mentions)
            observation_key = (*item["span"], notice_candidate)
            if observation_key in seen:
                continue
            seen.add(observation_key)
            common = {key: value for key, value in mentions.items() if key in refs}
            proof_only = bool(common) and _proof_only_notice(quote, common)
            phrases = []
            for premise in premises:
                for phrase in premise.get("association_phrases", []):
                    # A one-word adjective (e.g. finite) is not specific enough
                    # to associate an unnumbered paper-level correction.
                    words = re.findall(r"[A-Za-z]+", phrase.casefold())
                    if len(words) < 2:
                        continue
                    if re.search(r"\b" + r"[\s{}$()\\~-]+".join(map(re.escape, words)) + r"\b", quote, re.I):
                        phrases.append(phrase)
            strong_property = bool(phrases and _DOC_REVISION.search(quote) and not all_mentions)
            linked = bool(notice_candidate and associable and (common or strong_property))
            if common:
                basis = "explicit_statement_reference_in_supplied_revision_window"
            elif strong_property:
                basis = "explicit_revision_cue_and_matched_new_property_phrase_no_competing_reference"
            else:
                basis = "paper_level_correction_or_change_word_not_linked_to_this_statement"
            strength_scope = "proof_only_strength_unconfirmed" if proof_only else "added_premise_text_named" if strong_property or (common and phrases) else "statement_notice_strength_scope_unknown"
            finding = {"doc_id": doc_id, "source_type": source_type, "side": side, "effective_side": effective_side,
                "paper_id": document.get("paper_id"), "version": document.get("version"), "term": item["term"],
                "quote": quote[:800], "quote_total_characters": len(quote), "quote_truncated": len(quote) > 800,
                "span": item["span"], "cue_span": item["cue_span"],
                "span_unit": "unicode_codepoints_within_supplied_document_comment_masked_text_same_offsets",
                "source_coordinate_conversion": "not_performed", "association_basis": basis,
                "matched_statement_reference_keys": sorted(common), "matched_new_property_phrases": sorted(set(phrases)),
                "external_citation_reference_keys_excluded": sorted(all_mentions.keys() - mentions.keys()),
                "revision_notice_candidate": notice_candidate,
                "statement_association_established_by_text_rule": linked,
                "statement_notice_strength_scope": strength_scope,
                "explains_added_premise": bool(linked and not proof_only and (strong_property or phrases)),
                "target_version_relation": version_status, "historical_background_only": historical,
                "source_path": document.get("source_path"), "url": document.get("url"), "sha256": document.get("sha256"),
                "adequacy_of_explanation": "unknown", "mathematical_change_explained": "unknown"}
            finding["evidence_id"] = "disc_" + sha256((doc_id + repr(item["span"]) + basis).encode("utf-8")).hexdigest()[:24]
            if len(findings) < 40:
                findings.append(finding)
            if linked:
                statement_notices.append(finding)
            elif usable and notice_candidate:
                paper_notices.append(finding)
    if statement_notices:
        status, label = "statement_explanation_found", "给定材料找到与本条关联的修订说明；是否充分解释强度仍待核"
    elif paper_notices:
        status, label = "paper_correction_scope_unconfirmed", "本论文材料提到更正/变动，未确认覆盖本条"
    elif adequate:
        status, label = "not_found_in_checked_material", "已查给定材料范围内未找到本条说明；不能认证全渠道未说明"
    else:
        status, label = "material_missing", "缺少可归于后端论文/版本的独立说明材料；局部窗口不支持静默判断"
    selected = statement_notices + [item for item in paper_notices if item not in statement_notices]
    selected += [item for item in findings if item.get("historical_background_only") and item.get("revision_notice_candidate")]
    selected_ids = set()
    selected = [item for item in selected if not (item["evidence_id"] in selected_ids or selected_ids.add(item["evidence_id"]))][:24]
    coverage = {"scope": "only_supplied_evidence_documents_not_all_channels_or_all_historical_comments",
        "documents_provided": len(supplied), "documents_checked": len(coverage_docs),
        "document_limit": MAX_EVIDENCE_DOCUMENTS, "documents_omitted_by_limit": max(0, len(supplied) - MAX_EVIDENCE_DOCUMENTS),
        "characters_limit_per_document": MAX_EVIDENCE_TEXT, "documents": coverage_docs,
        "target_material_present": adequate,
        "full_document_checked": any(d["source_type"] == "full_tex" and d["eligible_for_target_notice_scan"] and not d["scan_truncated"] for d in coverage_docs),
        "source_types_checked": sorted({d["source_type"] for d in coverage_docs if d["eligible_for_target_notice_scan"]}),
        "historical_old_material_is_new_change_explanation": False,
        "all_historical_version_comments_checked": False, "independent_errata_search_exhaustive": False,
        "all_channel_silence_certified": False, "external_sources_opened_by_engine": False,
        "document_version_semantics": "version_is_target_paper_version_source_version_is_notice_own_version_not_compared",
        "statement_reference_keys": sorted(refs), "reference_numbering_is_tex_execution_certified": False}
    coverage["incidental_keyword_windows_excluded"] = sum(not item["revision_notice_candidate"] for item in findings)
    scopes = sorted({item["statement_notice_strength_scope"] for item in statement_notices})
    return {"status": status, "label": label, "matched_evidence": selected,
        "coverage": coverage, "silent_change": "unknown", "explanation_sufficiency": "unknown",
        "statement_notice_strength_scopes": scopes,
        "explains_added_premise": any(item["explains_added_premise"] for item in statement_notices),
        "runtime_inputs": "supplied_documents_and_statement_text_metadata_no_human_or_model_verdict"}, exported


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
