"""Shared text primitives: active-text masking, tokens, similarity, formula units and signal records."""

from __future__ import annotations

from collections import Counter
import re
import unicodedata


MAX_TEXT = 24000
MAX_CONTEXT = 12000
MAX_DISPLAY_TEXT = 9000
MAX_TOKENS = 5000
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
_CITATION = re.compile(r"\\(?:cite\w*|autocite|parencite|textcite)\*?(?:\s*\[[^\]]*\])*\s*\{([^{}]*)\}", re.I)
_STYLE = re.compile(r"\\(?:left|right|[bB]igg?[lrm]?|displaystyle|textstyle|quad|qquad|enspace|thinspace|notag|nonumber|mathstrut|strut)(?![A-Za-z])|\\[,;!:]")
_FONT_WRAPPER = re.compile(r"\\(?:bm|boldsymbol|mathbf|mathit|mathrm|mathsf|mathtt|mathnormal)\s*\{([^{}]*)\}")
_ENV = re.compile(r"\\(?:begin|end)\s*\{[^{}]*\}")


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
