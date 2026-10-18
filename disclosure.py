"""Revision notices, errata and full-TeX scans, and their association with the current statement."""

from __future__ import annotations

from hashlib import sha256
import re
from statement_text import MAX_CONTEXT, _BEGIN_DECL, _CITATION, _active_text, _paper_id, _quote, _string, _version


MAX_EVIDENCE_DOCUMENTS = 12
MAX_EVIDENCE_TEXT = 1000000
MAX_EVIDENCE_CACHE = 64
_CORRECTION = re.compile(
    r"\berrat(?:um|a)\b|\bcorrigend(?:um|a)\b|\bwithdrawn\b|\bincorrect\b|"
    r"\bcounter[-\s]?examples?\b|\bcorrection\b|\bcorrected\s+(?:version|statement|theorem|proof)\b|"
    r"\b(?:error|mistake)\s+in\b|\bdoes\s+not\s+hold\b|\bfalse\s+(?:statement|claim|theorem)\b|"
    r"\bshould\s+(?:instead\s+)?read\b|勘误|更正|反例|撤回|有误|不成立", re.I)
_CHANGE = re.compile(
    r"\brevis(?:ed|ion)\b|\breplac(?:ed|es)\b|\bweaken(?:ed|ing)\b|\bstronger\b|\bweaker\b|"
    r"\badditional\s+(?:hypothes[ei]s|assumptions?|conditions?)\b|\bnow\s+assume\b|修订|改为|新增假设|弱化", re.I)
_DOC_CHANGE = re.compile(r"\berrat(?:um|a)\b|\bcorrigend(?:um|a)\b|\b(?:mistake|incorrect|withdrawn|correction|corrected|replaced|revised|weakened|weaker|adjusted)\b|\b(?:missing|omitted|additional)\s+(?:assumption|hypothesis|condition)\b|\berror\s+in\b|勘误|更正|补充假设|漏掉.{0,8}假设|撤回", re.I)
_DOC_REVISION = re.compile(r"\berrat(?:um|a)\b|\bcorrigend(?:um|a)\b|\b(?:mistake|incorrect|withdrawn|corrected|correction|replaced|revised|weakened)\b|\b(?:missing|omitted|additional)\s+(?:assumption|hypothesis|condition)\b|\berror\s+in\b|勘误|更正|补充假设|漏掉.{0,8}假设|撤回", re.I)


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
