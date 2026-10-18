"""Statement alignment evidence and in-batch competition between candidate statements."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import re
from asymptotic_order import mask_bound_arguments
from statement_text import _BEGIN_DECL, _META, _RELATION, _TOKEN, _active_text, _formulas, _math_units, _normalized, _paper_id, _similarity, _tokens, _version


_STOP_ANCHORS = frozenset("let suppose assume assuming then for any all every some there exist exists have has the a an and or but by with without that this these those thus hence where which whose such be been is are was were will shall if only into from of to in on at as we it its their they our result theorem lemma proposition corollary proof follows holds show proves each respectively".split())


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
    # sides are mostly covered and the located conclusions nearly coincide,
    # ignoring the arguments of asymptotic bounds compared separately.
    bound_masked_similarity = _similarity(mask_bound_arguments(ac), mask_bound_arguments(bc)) if ac and bc else None
    short_match = (min(old_coverage, new_coverage) >= .75 and len(anchors) >= 2
                   and bound_masked_similarity is not None and bound_masked_similarity >= .9)
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
