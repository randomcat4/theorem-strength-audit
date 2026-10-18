"""Adversarial contract tests for deterministic investigation signals."""

from copy import deepcopy
import json
import unittest

from strength_signals import analyze_pairs


def side(paper, text, authors=None, version=None, date="", date_kind="version_date", context=""):
    return dict(paper_id=paper, text=text, authors=["A. Author"] if authors is None else authors,
                version=version, date=date, date_kind=date_kind, context=context,
                source={"path": "test-source.tex", "sha256": "test", "span": [0, len(text)], "source_available": True})


def pair(old=None, new=None, pair_id="test"):
    return dict(pair_id=pair_id,
                old=old or side("paper", "Let K/k be any field extension. Then F preserves products.", version=1),
                new=new or side("paper", "Let K/k be a finite field extension. Then F preserves products.", version=2),
                channels=["synthetic_unit_test"], evidence_kind="version_source")


class TestStrengthSignals(unittest.TestCase):
    def analyze(self, value):
        return analyze_pairs([value])[0]

    def kinds(self, record):
        return {s["kind"] for s in record["algorithm"]["signals"]}

    def test_reversed_versions_are_oriented_and_explained(self):
        p = pair()
        p["old"], p["new"] = p["new"], p["old"]
        r = self.analyze(p)
        self.assertEqual(r["old"]["version"], 1)
        self.assertEqual(r["new"]["version"], 2)
        self.assertEqual(r["algorithm"]["chronology"]["direction"], "reversed")
        self.assertTrue(r["algorithm"]["chronology"]["statement_order_confirmed"])
        self.assertIn("qualifier_change", self.kinds(r))

    def test_same_day_without_precise_versions_is_unknown(self):
        p = pair(side("a", "For all x, P(x).", date="2024-02-01"),
                 side("b", "For some x, P(x).", date="2024-02-01"))
        r = self.analyze(p)
        self.assertEqual(r["algorithm"]["chronology"]["direction"], "unknown")
        self.assertFalse(r["algorithm"]["chronology"]["statement_order_confirmed"])
        self.assertTrue(all(s["direction_hint"] == "unoriented_text_difference" for s in r["algorithm"]["signals"]))

    def test_first_submission_does_not_date_unknown_snapshot(self):
        p = pair(side("a", "For all x, P(x).", date="2020-01-01", date_kind="first_submission"),
                 side("b", "For some x, P(x).", date="2024-01-01", date_kind="first_submission"))
        r = self.analyze(p)
        self.assertEqual(r["algorithm"]["chronology"]["status"], "paper_date_only")
        self.assertFalse(r["algorithm"]["chronology"]["statement_order_confirmed"])
        self.assertTrue(any("首次上传" in w for w in r["algorithm"]["warnings"]))

    def test_same_day_precise_timestamps_can_sort(self):
        p = pair(side("a", "For all x, P(x).", date="2024-02-01T15:00:00Z"),
                 side("b", "For some x, P(x).", date="2024-02-01T10:00:00Z"))
        r = self.analyze(p)
        self.assertEqual(r["old"]["paper_id"], "b")
        self.assertTrue(r["algorithm"]["chronology"]["statement_order_confirmed"])

    def test_conflicting_dates_and_versions_do_not_choose_a_direction(self):
        p = pair(side("a", "For all x, P(x).", version=1, date="2025-01-01"),
                 side("a", "For some x, P(x).", version=2, date="2020-01-01"))
        r = self.analyze(p)
        self.assertEqual(r["algorithm"]["chronology"]["status"], "conflict")
        self.assertEqual(r["algorithm"]["chronology"]["direction"], "unknown")

    def test_matching_names_are_not_verified_identities(self):
        p = pair(side("a", "For all x, P(x).", authors=["J. Smith"], date="2020-01-01"),
                 side("b", "For some x, P(x).", authors=["J Smith"], date="2021-01-01"))
        r = self.analyze(p)
        self.assertEqual(r["algorithm"]["author_relation"]["kind"], "name_overlap")
        self.assertFalse(r["algorithm"]["author_relation"]["identity_verified"])

    def test_hint_cannot_promote_third_party_to_shared_author(self):
        p = pair(side("a", "For all x, P(x).", authors=["Alice"], date="2020-01-01"),
                 side("b", "For some x, P(x).", authors=["Bob"], date="2021-01-01"))
        p["author_match_hint"] = "same person maybe"
        r = self.analyze(p)
        self.assertEqual(r["algorithm"]["author_relation"]["kind"], "no_name_overlap")
        self.assertFalse(r["algorithm"]["author_relation"]["hint_used_for_identity"])

    def test_third_party_restricted_application_is_retained(self):
        p = pair(side("a", "Let K/k be any field extension. Then F preserves products.", authors=["Alice"], date="2020-01-01"),
                 side("b", "Let K/k be a finite field extension. Then F preserves products.", authors=["Bob"], date="2021-01-01", context="We only need the special case of the quoted theorem."))
        r = self.analyze(p)
        self.assertIn("restricted_use_context", self.kinds(r))
        self.assertIn("qualifier_change", self.kinds(r))
        self.assertGreater(r["algorithm"]["priority_score"], 0)
        self.assertFalse(r["algorithm"]["weaker_claim_certified"])

    def test_duplicates_do_not_inflate_paper_support_or_score(self):
        p = pair(side("a", "Let K/k be any field extension. Then F preserves products.", authors=["Alice"], date="2020-01-01"),
                 side("b", "Let K/k be a finite field extension. Then F preserves products.", authors=["Bob"], date="2021-01-01"))
        one = self.analyze(p)
        duplicate = deepcopy(p)
        duplicate["pair_id"] = "duplicate"
        many = analyze_pairs([p, duplicate, deepcopy(p)])
        self.assertEqual(many[0]["support"]["distinct_later_papers"], 1)
        self.assertEqual(many[0]["support"]["distinct_author_groups"], 1)
        self.assertEqual(many[0]["algorithm"]["priority_score"], one["algorithm"]["priority_score"])

    def test_multiple_third_party_users_aggregate_with_coauthor_groups(self):
        old = side("a", "Let K/k be any field extension. Then F preserves products.", authors=["Origin"], date="2020-01-01")
        text = "Let K/k be a finite field extension. Then F preserves products."
        values = [pair(old, side("b", text, authors=["Bob", "Carol"], date="2021-01-01"), "b"),
                  pair(old, side("c", text, authors=["Carol", "Dan"], date="2022-01-01"), "c"),
                  pair(old, side("d", text, authors=["Eve"], date="2023-01-01"), "d")]
        r = analyze_pairs(values)
        self.assertEqual(r[0]["support"]["distinct_later_papers"], 3)
        self.assertEqual(r[0]["support"]["distinct_author_groups"], 2)
        self.assertEqual(r[0]["support"]["nonoverlapping_author_groups"], 2)
        self.assertFalse(r[0]["support"]["corpus_prevalence_estimated"])

    def test_same_paper_versions_do_not_count_as_three_later_papers(self):
        p = pair()
        r = analyze_pairs([p, deepcopy(p)])
        self.assertEqual(r[0]["support"]["distinct_later_papers"], 0)
        self.assertEqual(r[0]["support"]["distinct_version_observations"], 1)

    def test_public_erratum_never_certifies_silence(self):
        p = pair()
        p["new"]["context"] = "Erratum. There was an error in the theorem. The corrected statement follows."
        r = self.analyze(p)
        self.assertEqual(r["algorithm"]["disclosure"]["status"], "matched_correction_language")
        self.assertFalse(r["algorithm"]["disclosure"]["silent_change_certified"])
        self.assertTrue(any(c["key"] == "correction_language" and c["points"] < 0 for c in r["algorithm"]["score_components"]))

    def test_no_keyword_is_only_a_window_nonmatch(self):
        r = self.analyze(pair())
        self.assertEqual(r["algorithm"]["disclosure"]["status"], "not_matched_in_provided_window")
        self.assertFalse(r["algorithm"]["disclosure"]["coverage"]["full_document_checked"])
        self.assertFalse(r["algorithm"]["disclosure"]["silent_change_certified"])

    def test_quantifiers_and_negation_are_not_removed(self):
        r = self.analyze(pair(side("a", r"For $\forall x$, $P(x)$ does not hold.", version=1),
                              side("a", r"For $\exists x$, $P(x)$ holds.", version=2)))
        self.assertIn("quantifier_change", self.kinds(r))
        self.assertIn("negation_change", self.kinds(r))
        self.assertFalse(r["algorithm"]["mathematical_entailment_checked"])

    def test_inequality_and_numeric_change_are_visible(self):
        r = self.analyze(pair(side("a", r"Let x be real. Then $f(x) \geq 4$.", version=1),
                              side("a", r"Let x be real. Then $f(x) \leq 2$.", version=2)))
        self.assertIn("inequality_change", self.kinds(r))
        signal = next(s for s in r["algorithm"]["signals"] if s["kind"] == "inequality_change")
        self.assertIn(r"\geq", signal["old_quote"])
        self.assertIn(r"\leq", signal["new_quote"])

    def test_infinite_is_not_finite(self):
        r = self.analyze(pair(side("a", "Let G be infinite. Then P(G) holds.", version=1),
                              side("a", "Let G be finite. Then P(G) holds.", version=2)))
        finite = [s for s in r["algorithm"]["signals"] if s.get("term") == "finite"]
        self.assertEqual(len(finite), 1)
        self.assertEqual(finite[0]["action"], "added")

    def test_length_and_similarity_do_not_alone_certify_or_score_weakening(self):
        p = pair(side("a", "Let X be a group. Then P(X) holds.", version=1),
                 side("a", "Let X be a group. Then P(X) holds. This is a rewording of our basic assertion.", version=2))
        r = self.analyze(p)
        self.assertLessEqual(r["algorithm"]["priority_score"], 8)
        self.assertNotIn("observable_text_changes", {v["key"] for v in r["algorithm"]["score_components"]})
        self.assertFalse(r["algorithm"]["weaker_claim_certified"])

    def test_first_statement_excludes_proof_and_later_theorem(self):
        a = r"\begin{theorem} Let X be a group. Then P(X).\begin{proof} All groups are finite.\end{proof}\end{theorem}\begin{theorem} Some Y is normal.\end{theorem}"
        b = r"\begin{theorem} Let X be a group. Then P(X).\begin{proof} No group is finite.\end{proof}\end{theorem}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["priority_score"], 0)
        self.assertNotIn("finite", r["old"]["text"])
        self.assertNotIn("Some Y", r["old"]["text"])
        self.assertTrue(any("声明提取" in w for w in r["algorithm"]["warnings"]))

    def test_source_ownership_uncertainty_is_not_a_false_statement_verdict(self):
        p = pair()
        p["old"]["source"].update(post_first_end_document=True, document_roots=2, preview_may_be_truncated=True)
        r = self.analyze(p)
        self.assertTrue(any("归属" in w or "所属" in w for w in r["algorithm"]["warnings"]))
        self.assertLessEqual(r["algorithm"]["priority_score"], 49)
        self.assertFalse(r["algorithm"]["weaker_claim_certified"])

    def test_deterministic_json_serializable_and_input_unchanged(self):
        p = pair()
        original = deepcopy(p)
        a = analyze_pairs(iter([p]))
        b = analyze_pairs(iter([p]))
        self.assertEqual(a, b)
        self.assertEqual(p, original)
        json.dumps(a, ensure_ascii=False)
        self.assertEqual(sum(c["points"] for c in a[0]["algorithm"]["score_components"]), a[0]["algorithm"]["priority_score"])

    def test_formula_sentence_punctuation_is_formatting_not_strength(self):
        a = r"Let x be real. Then $f(x) \geq 1$ and $g(x)=2$."
        b = r"Let x be real. Then $f(x) \geq 1,$ and $g(x)=2.$"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["priority_score"], 0)
        self.assertEqual(r["algorithm"]["candidate_state"], "formatting_observation")
        self.assertFalse(r["algorithm"]["eligible_for_strength_queue"])
        self.assertFalse(self.kinds(r) & {"formula_change", "numeric_formula_change", "inequality_change"})

    def test_formula_layout_commands_are_not_strength(self):
        a = r"Let x be real. Then $f(x)\geq 1$."
        b = r"Let x be real. Then $\displaystyle f\left(x\right)\geq 1\,.$"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["priority_score"], 0)
        self.assertEqual(r["algorithm"]["text_difference"]["status"], "formatting_only")

    def test_commented_formula_is_not_detected_or_quoted(self):
        a = "Let x be real. Then $f(x) \\geq 4$.\n% $evil(x) \\leq 666$\n"
        b = "Let x be real. Then $f(x) \\geq 2$.\n% $evil(x) \\leq 123$\n"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        formula = [s for s in r["algorithm"]["signals"] if s["kind"] in {"numeric_formula_change", "inequality_change", "formula_change"}]
        self.assertTrue(formula)
        self.assertTrue(all("evil" not in s["old_quote"] + s["new_quote"] for s in formula))
        self.assertTrue(all(s["quote_source"] == "comment_masked_active_statement" for s in formula))
        self.assertIn("4", formula[0]["old_quote"])
        self.assertIn("2", formula[0]["new_quote"])

    def test_comment_only_change_cannot_trigger_a_condition_or_erratum(self):
        a = "Let G be a group. Then P(G) holds.\n% normal finite erratum\n"
        b = "Let G be a group. Then P(G) holds.\n% no finite bound\n"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["priority_score"], 0)
        self.assertEqual(r["algorithm"]["disclosure"]["status"], "not_matched_in_provided_window")

    def test_macro_name_is_not_a_plain_property(self):
        a = r"Let X be a group. Then $\smooth{X}=0$."
        b = r"Let X be a group. Then $\regular{X}=0$."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        qualifiers = [s for s in r["algorithm"]["signals"] if s["kind"] == "qualifier_change"]
        self.assertFalse(any(s["term"] in {"smooth", "regular"} for s in qualifiers))

    def test_finite_in_compound_term_is_not_new_finiteness_premise(self):
        a = "Let F be a regular functor. Then F is an equivalence."
        b = "Let F be a fully faithful finite-limit-preserving functor. Then F is fully faithful."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        finite = [s for s in r["algorithm"]["signals"] if s.get("term") == "finite"]
        self.assertEqual(finite, [])
        self.assertTrue(any(s["kind"] == "terminology_change" and "finite-limit-preserving" in s["term"] for s in r["algorithm"]["signals"]))
        self.assertFalse(r["algorithm"]["eligible_for_strength_queue"])

    def test_different_named_results_cannot_get_high_priority_from_many_changes(self):
        a = r"\begin{theorem}[Optimal Smoother Equations] Let X be a smooth state process. Then $A(X)\geq 4$.\end{theorem}"
        b = r"\begin{theorem}[Optimal Filter Equations] Let X be a finite positive regular state process. Then $A(X)\leq 2$.\end{theorem}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["statement_alignment"], "named_statement_mismatch")
        self.assertLessEqual(r["algorithm"]["priority_score"], 24)
        self.assertFalse(r["algorithm"]["eligible_for_strength_queue"])
        self.assertEqual(r["algorithm"]["next_step"]["action"], "align_statements")

    def test_closer_competing_statement_blocks_unmatched_strength_queue(self):
        old = side("a", "Let G be any group. Then F(G) preserves products and exact sequences.", version=1)
        exact = side("a", "Let G be any group. Then F(G) preserves products and exact sequences.", version=2)
        different = side("a", "Let G be a finite group. Then F(G) preserves products and exact sequences.", version=2)
        r = analyze_pairs([pair(old, exact, "exact"), pair(old, different, "different")])
        self.assertEqual(r[1]["algorithm"]["statement_alignment"], "not_best_statement_candidate")
        self.assertFalse(r[1]["algorithm"]["eligible_for_strength_queue"])
        self.assertLessEqual(r[1]["algorithm"]["priority_score"], 39)

    def test_citation_separates_paper_authors_from_original_claim_ownership(self):
        p = pair()
        p["old"]["text"] = r"\begin{theorem}\cite[Theorem 1]{External} " + p["old"]["text"] + r"\end{theorem}"
        p["new"]["text"] = r"\begin{theorem}\cite[Theorem 2]{External} " + p["new"]["text"] + r"\end{theorem}"
        r = self.analyze(p)
        role = r["algorithm"]["assertion_role"]["old"]
        self.assertEqual(role["role"], "citation_in_statement")
        self.assertEqual(role["ownership_status"], "unknown")
        self.assertEqual(role["citation_keys"], ["External"])
        self.assertTrue(r["algorithm"]["eligible_for_self_revision_queue"])
        self.assertEqual(r["algorithm"]["self_revision_basis"]["assertion_original_ownership"], "unknown")
        self.assertTrue(any(v["key"] == "assertion_ownership_unknown" for v in r["algorithm"]["score_components"]))

    def test_additional_quantifier_scope_is_preserved_when_any_already_exists(self):
        a = r"For any X the following hold. \begin{enumerate}\item P(X).\item $Q_n$ fails.\end{enumerate}"
        b = r"For any X the following hold. \begin{enumerate}\item P(X).\item $Q_n$ fails for any $2<n\in\omega$.\end{enumerate}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["text_difference"]["status"], "active_text_changed")
        self.assertIn("formula_added", self.kinds(r))
        self.assertTrue(any(s.get("parameter_domain_candidate") for s in r["algorithm"]["signals"]))
        changes = r["algorithm"]["text_difference"]["changes"]
        self.assertTrue(any("2<n" in c["new_quote"] for c in changes))
        self.assertEqual(r["algorithm"]["decision_status"]["mathematical_weakening"], "unknown")
        self.assertGreater(r["algorithm"]["priority_score"], 0)

    def test_first_then_does_not_label_later_item_as_conclusion(self):
        a = r"Let X be a group. Then \begin{enumerate}\item P(X).\item Let Y be a group. Q(Y).\end{enumerate}"
        b = r"Let X be a group. Then \begin{enumerate}\item P(X).\item Let Y be a finite group. Q(Y).\end{enumerate}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        finite = next(s for s in r["algorithm"]["signals"] if s.get("term") == "finite")
        self.assertEqual(finite["scope"], "premise_like_text")
        self.assertEqual(finite["logical_role_status"], "unknown")

    def test_dependency_not_is_not_presented_as_property_negation(self):
        a = "Let x be real. Then $C(x)=1$, where C does not depend on j."
        b = "Let x be real. Then $C(x)=1$, where C may depend on j."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertIn("dependency_phrase_change", self.kinds(r))
        self.assertNotIn("negation_change", self.kinds(r))

    def test_support_counts_have_nonadditive_group_contract(self):
        p = pair(side("a", "Let K/k be any field extension. Then F preserves products.", authors=["Alice"], date="2020-01-01"),
                 side("b", "Let K/k be a finite field extension. Then F preserves products.", authors=["Bob"], date="2021-01-01"))
        r = analyze_pairs([p, deepcopy(p)])
        support = r[0]["support"]
        self.assertFalse(support["add_across_rows"])
        self.assertEqual(support["deduplicate_across_rows_by"], "group_key")
        self.assertEqual(support["third_party_name_later_paper_ids"], ["b"])
        self.assertEqual(support["author_groups"][0]["paper_ids"], ["b"])
        self.assertFalse(support["author_groups"][0]["overlaps_old_names"])
        self.assertEqual(r[0]["support"]["group_key"], r[1]["support"]["group_key"])

    def test_version_author_rosters_are_not_certified_from_corpus_metadata(self):
        r = self.analyze(pair())
        self.assertFalse(r["algorithm"]["author_relation"]["historical_version_author_lists_verified"])
        self.assertEqual(r["algorithm"]["decision_status"]["original_claim_ownership"], "unknown")
        self.assertFalse(r["algorithm"]["weaker_claim_certified"])

    def test_factoring_and_inequality_are_not_erased_as_sentence_punctuation(self):
        a = r"Let n be positive. Then $f(n)=n!$ and $g(n)\not\leq 1$."
        b = r"Let n be positive. Then $f(n)=n$ and $g(n)\geq 1$."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["text_difference"]["status"], "active_text_changed")
        self.assertTrue(self.kinds(r) & {"formula_change", "negation_change", "inequality_change"})
        self.assertNotEqual(r["algorithm"]["candidate_state"], "formatting_observation")

    def test_factorial_alone_is_not_erased_as_exclamation_punctuation(self):
        a = r"Let n be a positive integer. Then $f(n)=n!$."
        b = r"Let n be a positive integer. Then $f(n)=n$."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["text_difference"]["status"], "active_text_changed")
        self.assertNotEqual(r["algorithm"]["priority_score"], 0)

    def test_internal_formula_comma_is_not_erased_as_sentence_punctuation(self):
        a = r"Let a and b be real. Then $F(a,b)=1$."
        b = r"Let a and b be real. Then $F(a b)=1$."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["text_difference"]["status"], "active_text_changed")
        self.assertTrue(r["algorithm"]["text_difference"]["changes"])

    def test_real_aligned_restriction_still_enters_queue_without_manual_unlock(self):
        r = self.analyze(pair())
        self.assertTrue(r["algorithm"]["eligible_for_strength_queue"])
        self.assertEqual(r["algorithm"]["candidate_state"], "strength_text_candidate")
        self.assertGreaterEqual(r["algorithm"]["priority_score"], 65)
        self.assertEqual(r["algorithm"]["decision_status"]["mathematical_weakening"], "unknown")

    def test_verified_index_text_does_not_need_whole_tex_cache_to_yield_hint(self):
        p = pair(side("a", "Let K/k be any field extension. Then F preserves products.", authors=["Alice"], date="2020-01-01", date_kind="first_submission"),
                 side("b", "Let K/k be a finite field extension. Then F preserves products.", authors=["Alice"], date="2021-01-01", date_kind="first_submission"))
        p["new"]["source"].update(source_available=False, hydration_status="VERIFIED_INDEX_ID_AND_SPAN")
        r = self.analyze(p)
        self.assertEqual(r["algorithm"]["chronology"]["status"], "paper_date_only")
        self.assertTrue(r["algorithm"]["eligible_for_strength_queue"])
        self.assertNotEqual(r["algorithm"]["candidate_state"], "source_unverified")

    def test_single_formula_to_multiple_branches_preserves_changed_row(self):
        premise = "Let X be a compact metric space and let n be a positive integer. Assume the regularity bound on the metric measure is uniform. Then "
        a = premise + r"$$F(X,n)\geq 6/n.$$"
        b = premise + r"\begin{align*}F(X,n)&\geq 6/n &\text{if $n\leq 10$,}\\ F(X,n)&\geq 3/n &\text{if $10<n\leq 20$,}\\ F(X,n)&\geq 2/n &\text{if $n>20$.}\end{align*}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        signals = [s for s in r["algorithm"]["signals"] if s["kind"] == "numeric_formula_change"]
        self.assertTrue(signals)
        self.assertIn("6/n", signals[0]["old_quote"])
        self.assertNotIn("6/n", signals[0]["new_quote"])
        self.assertIn(signals[0]["new_comparison_object"], ("F(X,n)", "f(x,n)"))
        self.assertTrue(signals[0]["new_branch_condition"])
        self.assertEqual(signals[0]["formula_change_structure"], "single_to_branches")
        self.assertEqual(r["algorithm"]["statement_alignment"], "branch_expansion_textual_match")
        self.assertTrue(r["algorithm"]["eligible_for_strength_queue"])
        self.assertEqual(r["algorithm"]["decision_status"]["mathematical_weakening"], "unknown")

    def test_equation_align_and_dollar_environment_rewrite_is_formatting_only(self):
        a = r"Let X be a compact metric space. Then $$F(X)\geq 6/n.$$"
        b = r"Let X be a compact metric space. Then \begin{align*} F(X)&\geq 6/n\notag\label{renumbered}\end{align*}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["priority_score"], 0)
        self.assertEqual(r["algorithm"]["text_difference"]["status"], "formatting_only")
        self.assertFalse(r["algorithm"]["eligible_for_strength_queue"])
        self.assertFalse(self.kinds(r) & {"formula_added", "formula_omitted", "numeric_formula_change"})

    def test_typographical_continuation_row_is_not_a_new_branch(self):
        a = r"Let X be a compact metric space. Then $$F(X)=A(X)+C(X).$$"
        b = r"Let X be a compact metric space. Then \begin{align*}F(X)&=A(X)\\[2pt]&+C(X).\end{align*}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["priority_score"], 0)
        self.assertFalse(r["algorithm"]["alignment_evidence"]["branch_expansion_candidate"])

    def test_cases_inherit_object_and_show_actual_modified_branch(self):
        premise = "Let x be real and let the continuous function F satisfy the uniform regularity assumptions. Then "
        a = premise + r"$$F(x)\geq 1.$$"
        b = premise + r"\begin{equation*}F(x)\geq\begin{cases}1 & x\leq 0,\\1/2 & x>0.\end{cases}\end{equation*}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        signals = [s for s in r["algorithm"]["signals"] if s["kind"] == "numeric_formula_change"]
        self.assertTrue(signals)
        self.assertIn("1/2", signals[0]["new_quote"])
        self.assertTrue(signals[0]["new_comparison_object_inherited"])
        self.assertEqual(signals[0]["old_comparison_object"], signals[0]["new_comparison_object"])
        self.assertTrue(r["algorithm"]["eligible_for_strength_queue"])

    def test_same_premise_but_different_comparison_object_remains_alignment_pending(self):
        premise = "Let X be a compact metric space and let n be a positive integer. Assume the regularity bound on the metric measure is uniform. Then "
        a = premise + r"$$F(X,n)\geq 6/n.$$"
        b = premise + r"\begin{align*}G(X,n)&\geq 6/n &\text{if $n\leq 10$,}\\G(X,n)&\geq 3/n &\text{if $n>10$.}\end{align*}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["statement_alignment"], "comparison_object_mismatch")
        self.assertFalse(r["algorithm"]["alignment_evidence"]["branch_expansion_candidate"])
        self.assertFalse(r["algorithm"]["eligible_for_strength_queue"])
        self.assertLessEqual(r["algorithm"]["priority_score"], 24)

    def test_math_environments_in_comments_cannot_supply_branch_evidence(self):
        a = "Let X be a compact metric space. Then $F(X)=1$.\n% \\begin{align}F(X)=999\\\\F(X)=888\\end{align}\n"
        b = "Let X be a compact metric space. Then $F(X)=2$.\n% \\begin{align}F(X)=777\\\\F(X)=666\\end{align}\n"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertFalse(r["algorithm"]["alignment_evidence"]["branch_expansion_candidate"])
        for signal in r["algorithm"]["signals"]:
            self.assertNotIn("999", signal["old_quote"])
            self.assertNotIn("777", signal["new_quote"])

    def test_single_sided_parameter_domain_is_named_and_kept(self):
        a = r"For any X the following hold. \begin{enumerate}\item The regular construction is defined.\item $Q_n$ fails in $N$.\end{enumerate}"
        b = r"For any X the following hold. \begin{enumerate}\item The regular construction is defined.\item $Q_n$ fails in $N$ for any $2<n\in\omega$.\end{enumerate}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        added = [s for s in r["algorithm"]["signals"] if s["kind"] == "formula_added"]
        self.assertTrue(added)
        self.assertIn("2<n", added[0]["new_quote"])
        self.assertTrue(added[0]["parameter_domain_candidate"])
        self.assertTrue(r["algorithm"]["eligible_for_strength_queue"])
        self.assertEqual(added[0]["direction_hint"], "unresolved_strength_direction")

    def test_single_sided_formula_omission_is_kept_without_certifying_relaxation(self):
        premise = "Let X be a compact metric space and F a regular continuous operator. "
        a = premise + r"For any $n\geq 2$, the stable construction exists."
        b = premise + "For any n, the stable construction exists."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertIn("formula_omitted", self.kinds(r))
        self.assertTrue(r["algorithm"]["eligible_for_strength_queue"])
        self.assertFalse(r["algorithm"]["weaker_claim_certified"])

    def test_equal_core_formula_with_new_branch_scope_has_a_named_signal(self):
        premise = "Let X be a compact metric space and let n be a positive integer. Assume the regularity bound on the metric measure is uniform. Then "
        a = premise + r"$$F(X,n)\geq 6/n.$$"
        b = premise + r"\begin{align*}F(X,n)&\geq 6/n &\text{if $n\leq 10$,}\\F(X,n)&\geq 6/n &\text{if $n>10$.}\end{align*}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertIn("branch_scope_change", self.kinds(r))
        self.assertNotIn("numeric_formula_change", self.kinds(r))
        self.assertTrue(r["algorithm"]["eligible_for_strength_queue"])
        self.assertEqual(r["algorithm"]["decision_status"]["mathematical_weakening"], "unknown")

    def test_raw_observations_remain_when_alignment_gate_qualifies_none(self):
        old = side("a", r"\begin{theorem}[Smoother Formula] Let X be a compact metric space. Then $F(X)\geq 4$.\end{theorem}", date="2020-01-01")
        new = side("b", r"\begin{theorem}[Filter Formula] Let X be a finite compact metric space. Then $G(X)\geq 2$.\end{theorem}", date="2021-01-01")
        r = self.analyze(pair(old, new))
        self.assertEqual(r["support"]["raw_observed_distinct_later_papers"], 1)
        self.assertEqual(r["support"]["raw_observed_later_paper_ids"], ["b"])
        self.assertEqual(r["support"]["qualified_distinct_later_papers"], 0)
        self.assertIsNone(r["support"]["verified_adoption"])

    def test_normal_word_is_not_translated_into_one_mathematical_meaning(self):
        a = "Let X have a distribution. Then F(X) has finite expectation."
        b = "Let X have a normal distribution. Then F(X) has finite expectation."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        normal = next(s for s in r["algorithm"]["signals"] if s.get("term") == "normal")
        self.assertIn("normal", normal["label"])
        self.assertNotIn("正规", normal["label"])
        self.assertNotIn("正态", normal["label"])

    def test_footnote_citation_scope_does_not_certify_whole_statement_origin(self):
        p = pair()
        p["old"]["text"] += r" A term is defined in a footnote\footnote{See \cite{DefinitionSource} for notation.}."
        p["new"]["text"] += r" A term is defined in a footnote\footnote{See \cite{DefinitionSource} for notation.}."
        r = self.analyze(p)
        role = r["algorithm"]["assertion_role"]["old"]
        self.assertIn("inside_footnote", role["citation_locations"])
        self.assertEqual(role["citation_scope_status"], "unknown")
        self.assertFalse(role["opening_attribution_cue"])
        self.assertTrue(r["algorithm"]["eligible_for_self_revision_queue"])
        self.assertEqual(r["algorithm"]["self_revision_basis"]["status"], "investigation_route_only")
        self.assertEqual(r["algorithm"]["self_revision_basis"]["assertion_original_ownership"], "unknown")

    def test_competing_statement_can_be_located_by_pair_and_statement_id(self):
        old = side("a", "Let G be any group. Then F(G) preserves products and exact sequences.", version=1)
        exact = side("a", "Let G be any group. Then F(G) preserves products and exact sequences.", version=2)
        exact["statement_id"] = "exact_statement"
        different = side("a", "Let G be a finite group. Then F(G) preserves products and exact sequences.", version=2)
        different["statement_id"] = "alternative_statement"
        r = analyze_pairs([pair(old, exact, "exact_pair"), pair(old, different, "different_pair")])
        alternative = r[1]["algorithm"]["alignment_evidence"]["competition"]["forward"]["best_alternative"]
        self.assertEqual(alternative["pair_id"], "exact_pair")
        self.assertEqual(alternative["statement_id"], "exact_statement")

    def test_signal_identifiers_and_positions_are_deterministic(self):
        a, b = self.analyze(pair()), self.analyze(pair())
        self.assertTrue(a["algorithm"]["signals"])
        self.assertEqual([s["signal_id"] for s in a["algorithm"]["signals"]],
                         [s["signal_id"] for s in b["algorithm"]["signals"]])
        self.assertTrue(all(s["event_id"] == s["signal_id"] for s in a["algorithm"]["signals"]))

    def test_bracket_size_commands_are_formatting_not_formula_events(self):
        a = r"Let X be a compact metric space. Then $F(X)=N_l(m(X),R(X))$."
        b = r"Let X be a compact metric space. Then $F(X)=N_l\bigl(m(X),R(X)\bigr)$."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["text_difference"]["status"], "formatting_only")
        self.assertFalse(r["algorithm"]["eligible_for_strength_queue"])
        self.assertFalse(self.kinds(r) & {"formula_change", "formula_added", "formula_omitted"})
        self.assertEqual(r["algorithm"]["priority_score"], 0)

    def test_no_shared_textual_object_produces_unpaired_observations(self):
        a = r"Let X be a compact metric space. Then $A_n$ is defined."
        b = r"Let X be a compact metric space. Then $B_{n+1}=C_n+D_n$ is defined."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        formulas = [s for s in r["algorithm"]["signals"] if s["kind"].startswith("formula_")]
        self.assertTrue(formulas)
        self.assertFalse(self.kinds(r) & {"numeric_formula_change", "inequality_change", "formula_change"})
        self.assertTrue(all(s["pairing_status"] == "unpaired_fragment" for s in formulas))
        self.assertTrue(all("formula_change_structure" not in s for s in formulas))

    def test_conditional_inline_formula_quotes_keep_or_then_and_common_consequence(self):
        a = r"Let the regular operator F be given. If $\psi^*$ is trivial, then assume also $|t_\psi|>1$. The result holds."
        b = r"Let the regular operator F be given. If $R\otimes\psi^*=R$ or $S\otimes\psi^*=S$, then assume also $|t_\psi|>1$. The result holds."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        old = next(s for s in r["algorithm"]["signals"] if s["kind"] == "formula_omitted")
        new = next(s for s in r["algorithm"]["signals"] if s["kind"] == "formula_added")
        self.assertIn("is trivial", old["old_quote"])
        self.assertIn("then assume also", old["old_quote"])
        self.assertIn(r"|t_\psi|>1", old["old_quote"])
        self.assertIn(" or ", new["new_quote"])
        self.assertIn("then assume also", new["new_quote"])
        for s in (old, new):
            key = s["changed_side"]
            start, end = s["changed_span"]
            self.assertEqual(r[key]["text"][start:end], s[key + "_quote"])
            self.assertEqual(s["span_unit"], "characters_within_extracted_statement")
        self.assertEqual(len({s["event_id"] for s in r["algorithm"]["signals"]}), len(r["algorithm"]["signals"]))

    def test_removed_explicit_formula_list_shows_matching_label_backreference(self):
        a = r"Let F and G be regular functions. Assume that \begin{equation}\label{display_functions} L(s,F),L(s,G)\end{equation} are entire. Then P(F) holds."
        b = r"Let F and G be regular functions. Assume that the functions in \eqref{display_functions} are entire. Then P(F) holds."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        old = next(s for s in r["algorithm"]["signals"] if s["kind"] == "formula_omitted")
        self.assertIn(r"\eqref{display_functions}", old["new_quote"])
        self.assertIn("are entire", old["new_quote"])
        self.assertEqual(old["label_backreference"]["label"], "display_functions")
        self.assertEqual(old["pairing_status"], "unpaired_fragment")

    def test_discussing_a_conjecture_does_not_reclassify_a_theorem(self):
        a = r"\begin{theorem} Let G fulfill the spectral conjecture. Then F preserves products and exact sequences.\end{theorem}"
        b = r"\begin{theorem} Let G be finite and fulfill the spectral conjecture. Then F preserves products and exact sequences.\end{theorem}"
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertNotEqual(r["algorithm"]["assertion_role"]["old"]["role"], "conjecture_cue")
        self.assertNotEqual(r["algorithm"]["assertion_role"]["new"]["role"], "conjecture_cue")

    def test_explicit_conjecture_metadata_is_available_without_an_environment(self):
        p = pair()
        p["old"]["source"]["declaration_kind"] = "conjecture"
        p["new"]["source"]["declaration_kind"] = "conjecture"
        r = self.analyze(p)
        self.assertEqual(r["algorithm"]["assertion_role"]["old"]["role"], "conjecture_cue")
        self.assertEqual(r["algorithm"]["assertion_role"]["old"]["declaration_kind_provided"], "conjecture")

    def test_definition_metadata_keeps_author_text_revision_route_open(self):
        p = pair()
        p["old"]["source"]["declaration_kind"] = "definition"
        p["new"]["source"]["declaration_kind"] = "definition"
        r = self.analyze(p)
        self.assertEqual(r["algorithm"]["assertion_role"]["old"]["role"], "definition_cue")
        self.assertTrue(r["algorithm"]["eligible_for_self_revision_queue"])
        self.assertEqual(r["algorithm"]["self_revision_basis"]["assertion_original_ownership"], "unknown")

    def test_bold_wrapper_is_a_notation_observation_not_certified_format_equivalence(self):
        a = r"Let X be a compact metric space. Then F(X) is $1$-conservative."
        b = r"Let X be a compact metric space. Then F(X) is $\bm{1}$-conservative."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertEqual(r["algorithm"]["text_difference"]["status"], "notation_only")
        self.assertEqual(r["algorithm"]["candidate_state"], "notation_observation")
        self.assertFalse(r["algorithm"]["eligible_for_strength_queue"])
        self.assertIn("notation_change", self.kinds(r))
        self.assertFalse(self.kinds(r) & {"formula_added", "formula_omitted"})
        self.assertEqual(r["algorithm"]["decision_status"]["mathematical_weakening"], "unknown")

    def test_notation_observation_does_not_override_real_condition_change(self):
        a = r"Let K/k be any field extension. Then F preserves products and is $1$-conservative."
        b = r"Let K/k be a finite field extension. Then F preserves products and is $\bm{1}$-conservative."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        self.assertIn("notation_change", self.kinds(r))
        self.assertIn("qualifier_change", self.kinds(r))
        self.assertEqual(r["algorithm"]["text_difference"]["status"], "active_text_changed")
        self.assertTrue(r["algorithm"]["eligible_for_strength_queue"])

    def test_long_changed_text_has_explicit_truncation_and_coordinate_origin(self):
        oldtext = "   \\begin{theorem} Let X be a regular object. " + "a" * 700 + ".\\end{theorem}  \n"
        newtext = "   \\begin{theorem} Let X be a regular object. " + "b" * 800 + ".\\end{theorem}  \n"
        p = pair(side("a", oldtext, version=1), side("a", newtext, version=2))
        p["old"]["source"]["span"] = [900000, 900000 + len(oldtext)]
        r = self.analyze(p)
        change = r["algorithm"]["text_difference"]["changes"][0]
        self.assertTrue(change["old_changed_text_truncated"])
        self.assertTrue(change["new_changed_text_truncated"])
        self.assertEqual(len(change["old_changed_text"]), 500)
        self.assertEqual(change["old_changed_text_total_characters"], 700)
        self.assertEqual(change["new_changed_text_total_characters"], 800)
        for key, original in (("old", oldtext), ("new", newtext)):
            cut = r[key]["statement_slice"]
            self.assertEqual(original[cut["start"]:cut["end"]], r[key]["text"])
            self.assertEqual(cut["end"] - cut["start"], len(r[key]["text"]))
            self.assertEqual(r[key]["statement_coordinates"]["source_coordinate_conversion"], "not_performed")
        self.assertEqual(r["algorithm"]["text_difference"]["coordinate_system"]["base"], "extracted_statement_before_display_truncation")
        self.assertLess(change["old_span"][0], 900000)

    def test_condition_word_quote_starts_before_the_triggering_if_clause(self):
        prefix = "If the initial covariance matrix " + "has the stated regularity and symmetry and " * 4
        a = prefix + "is definite, then all matrices remain definite."
        b = prefix + "is positive-definite, then all matrices remain positive-definite."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        positive = next(s for s in r["algorithm"]["signals"] if s.get("term") == "positive-definite")
        self.assertIn("If the initial covariance matrix", positive["new_quote"])
        self.assertIn("then all matrices", positive["new_quote"])

    def test_condition_trigger_is_prioritized_over_a_preceding_displayed_list(self):
        a = r"Assume the functions \begin{equation}\label{Flist} A(s),B(s).\end{equation} If $\psi^*$ is trivial, then assume also $|t_\psi|>1$."
        b = r"Assume the functions in \eqref{Flist} are entire. If $R\otimes\psi^*=R$ or $S\otimes\psi^*=S$, then assume also $|t_\psi|>1$."
        r = self.analyze(pair(side("a", a, version=1), side("a", b, version=2)))
        unpaired = [s for s in r["algorithm"]["signals"] if s["kind"] in {"formula_added", "formula_omitted"}]
        self.assertTrue(unpaired[0]["new_quote"].startswith("If "))
        self.assertTrue(unpaired[1]["old_quote"].startswith("If "))
        self.assertIn("is trivial", unpaired[1]["old_quote"])
        self.assertIn("then assume also", unpaired[1]["old_quote"])


def document(text, source_type="full_tex", paper_id="paper", version=2, **extra):
    return dict(text=text, source_type=source_type, paper_id=paper_id, version=version,
                source_available=True, side="new", source_path="supplied-not-opened.tex",
                sha256="container-file-hash", coverage_scope="supplied_complete_text",
                identity_basis="supplied_paper_version_metadata", association_scope="paper", **extra)


class TestAlgorithm030(unittest.TestCase):
    def analyze(self, value):
        return analyze_pairs([value])[0]

    def rewrite(self, old, new):
        return self.analyze(pair(side("paper", old, version=1), side("paper", new, version=2)))["algorithm"]

    def qualified_pair(self):
        p = pair()
        p["evidence_documents"] = [document("The finite extension construction and its proof are presented below.")]
        return p

    def test_explicit_equivalent_conditions_can_be_permuted(self):
        prefix = r"Let X be a compact metric space. The following conditions are equivalent: "
        a = prefix + r"\begin{enumerate}\item $F(X)=0$.\item $H(X)=0$.\end{enumerate}"
        b = prefix + r"\begin{enumerate}\item $H(X)=0$.\item $F(X)=0$.\end{enumerate}"
        r = self.rewrite(a, b)
        self.assertEqual(r["rewrite_equivalence"]["status"], "recognized_rewrite_only")
        self.assertFalse(r["eligible_for_strength_queue"])
        self.assertEqual(r["investigation_queue"]["status"], "rewrite_only")
        self.assertEqual(r["priority_score"], 0)
        self.assertEqual(sum(c["points"] for c in r["score_components"]), 0)

    def test_equivalent_list_inline_and_itemize_keep_all_clauses(self):
        a = r"For any path-connected space $Z$, $z_0\in Z$, the following statements are equivalent: \begin{itemize}\item $F(Z,z_0)$ is trivial;\item $H_2(Z;\mathbb Z)$ is trivial;\item $G(Z,z_0)$ is trivial.\end{itemize}"
        b = r"For any path-connected space $Z$ and any point $z_0\in Z$, the following statements are equivalent: \\ $({\rm i})$ $G(Z,z_0)$ is trivial;\\ $({\rm ii})$ $F(Z,z_0)$ is trivial; and\\ $({\rm iii})$ $H_2(Z;\mathbb{Z})$ is trivial."
        r = self.rewrite(a, b)
        self.assertTrue(r["rewrite_equivalence"]["suppresses_strength_alert"])

    def test_ordinary_ordered_list_is_not_permutation_equivalent(self):
        a = r"Let X be a regular object. Perform these steps: \begin{enumerate}\item Apply F to X.\item Apply H to X.\end{enumerate}"
        b = r"Let X be a regular object. Perform these steps: \begin{enumerate}\item Apply H to X.\item Apply F to X.\end{enumerate}"
        self.assertFalse(self.rewrite(a, b)["rewrite_equivalence"]["suppresses_strength_alert"])

    def test_equivalent_list_duplicate_and_negation_are_not_discarded(self):
        prefix = r"The following statements are equivalent: \begin{itemize}"
        a = prefix + r"\item $P(X)$ holds.\item $Q(X)$ holds.\end{itemize}"
        variants = [prefix + r"\item $Q(X)$ holds.\item $P(X)$ holds.\item $P(X)$ holds.\end{itemize}",
                    prefix + r"\item $Q(X)$ does not hold.\item $P(X)$ holds.\end{itemize}"]
        for b in variants:
            with self.subTest(new=b):
                self.assertFalse(self.rewrite(a, b)["rewrite_equivalence"]["suppresses_strength_alert"])

    def test_equivalent_list_direction_parameter_and_case_are_preserved(self):
        prefix = r"The following statements are equivalent: \begin{itemize}"
        a = prefix + r"\item $F(A)\geq n$.\item $Q(A)=0$.\end{itemize}"
        for b in (prefix + r"\item $Q(A)=0$.\item $F(A)\leq n$.\end{itemize}",
                  prefix + r"\item $Q(A)=0$.\item $F(A)\geq m$.\end{itemize}",
                  prefix + r"\item $Q(a)=0$.\item $F(a)\geq n$.\end{itemize}"):
            with self.subTest(new=b):
                self.assertFalse(self.rewrite(a, b)["rewrite_equivalence"]["suppresses_strength_alert"])

    def test_nested_lists_and_order_references_decline_rewrite_filter(self):
        a = r"The following conditions are equivalent: \begin{itemize}\item P holds.\item The preceding condition holds.\end{itemize}"
        b = r"The following conditions are equivalent: \begin{itemize}\item The preceding condition holds.\item P holds.\end{itemize}"
        self.assertFalse(self.rewrite(a, b)["rewrite_equivalence"]["suppresses_strength_alert"])
        a = r"The following conditions are equivalent: \begin{itemize}\item P holds.\begin{itemize}\item R holds.\end{itemize}\item Q holds.\end{itemize}"
        b = a.replace("P holds", "Q holds").replace("Q holds.\\end", "P holds.\\end")
        self.assertFalse(self.rewrite(a, b)["rewrite_equivalence"]["suppresses_strength_alert"])

    def test_custom_mathematical_item_labels_are_not_dropped(self):
        a = r"The following conditions are equivalent: \begin{itemize}\item[$n=1$] P holds.\item[$n=2$] Q holds.\end{itemize}"
        b = r"The following conditions are equivalent: \begin{itemize}\item[$n=3$] Q holds.\item[$n=1$] P holds.\end{itemize}"
        self.assertFalse(self.rewrite(a, b)["rewrite_equivalence"]["suppresses_strength_alert"])

    def test_isomorphic_symbol_and_explicit_words_are_supported_only_as_rewrite(self):
        a = r"Let X be a finite regular object. If $X\otimes X^*\cong T$, then F(X) preserves products."
        b = r"Let X be a finite regular object. If $X\otimes X^*$ isomorphic to $T$, then F(X) preserves products."
        r = self.rewrite(a, b)
        self.assertTrue(r["rewrite_equivalence"]["suppresses_strength_alert"])
        self.assertFalse(r["eligible_for_primary_investigation_queue"])
        self.assertEqual(r["decision_status"]["mathematical_weakening"], "unknown")
        b = b.replace("isomorphic to", "not isomorphic to")
        self.assertFalse(self.rewrite(a, b)["rewrite_equivalence"]["suppresses_strength_alert"])

    def test_symbol_dictionary_does_not_remove_reference_changes(self):
        a = r"If $X\cong T$, then use \ref{first_result}."
        b = r"If $X$ isomorphic to $T$, then use \ref{second_result}."
        self.assertFalse(self.rewrite(a, b)["rewrite_equivalence"]["suppresses_strength_alert"])

    def test_for_any_to_let_is_not_a_new_premise_by_itself(self):
        a = r"For any Noetherian local ring $(A,m)$ the construction is regular. Then F(A) preserves products."
        b = r"Let $(A,m)$ be a Noetherian local ring. The construction is regular. Then F(A) preserves products."
        r = self.rewrite(a, b)
        self.assertEqual(r["new_premises"], [])
        self.assertNotEqual(r["investigation_queue"]["status"], "primary_unexplained_tightening")

    def test_ring_to_domain_matches_same_declared_object_and_quote_coordinates(self):
        a = r"For any Noetherian local ring $(A,m)$ the regular construction is defined. Then F(A) preserves products and exact sequences."
        b = r"Let $(A,m)$ be a Noetherian local domain. The regular construction is defined. Then F(A) preserves products and exact sequences."
        r = self.analyze(pair(side("paper", a, version=1), side("paper", b, version=2)))
        premise = next(p for p in r["algorithm"]["new_premises"] if p["property"] == "integral_domain")
        self.assertEqual(premise["object_key"], "(A,m)")
        for key in ("old", "new"):
            start, end = premise[key + "_span"]
            self.assertEqual(r[key]["text"][start:end], premise[key + "_minimal"])
        self.assertEqual(premise["logical_strength"], "unknown")

    def test_new_property_on_different_object_does_not_pass_object_gate(self):
        a = r"Let $A$ be a ring. Then F(A) preserves products and exact sequences."
        b = r"Let $B$ be an integral domain. Then F(A) preserves products and exact sequences."
        self.assertEqual(self.rewrite(a, b)["new_premises"], [])

    def test_nonzero_solution_and_mixed_conclusion_remain_textual(self):
        a = r"Let $c$ be an arbitrary solution of the regular system. Then $F(c)=H_1(c)$."
        b = r"Let $c$ be an arbitrary nonzero solution of the regular system. Then $F(c)=H_2(c)$."
        r = self.rewrite(a, b)
        self.assertEqual([p["property"] for p in r["new_premises"]], ["nonzero"])
        self.assertEqual(r["tightening_textual_evidence"]["status"], "aligned_new_premises_with_other_changes")
        self.assertEqual(r["tightening_textual_evidence"]["logical_weakening"], "unknown")

    def test_negated_or_alternative_property_is_not_each_objects_new_attribute(self):
        a = r"Let $c$ be a solution of the regular system. Then F(c) preserves products."
        for description in ("not a nonzero solution", "a solution with at least one nonzero component"):
            b = "Let $c$ be " + description + ". Then F(c) preserves products."
            with self.subTest(description=description):
                self.assertNotIn("nonzero", [p["property"] for p in self.rewrite(a, b)["new_premises"]])

    def test_rational_line_requires_existing_same_parameter(self):
        a = r"The dimension of the orbit through $\lambda$ is at least $F(G,\lambda)=2r+1$."
        b = r"The dimension of the orbit through $\lambda$ lying on some rational line in $V^*$ is at least $F(G,\lambda)=2r+1$."
        r = self.rewrite(a, b)
        self.assertIn("rational_line", [p["property"] for p in r["new_premises"]])
        self.assertEqual(r["tightening_textual_evidence"]["conclusion_match"]["status"], "shared_guarantee_formula")
        b = b.replace(r"through $\lambda$", r"through $\mu$")
        self.assertEqual(self.rewrite(a, b)["new_premises"], [])

    def test_dense_image_predicate_only_in_premise_syntax(self):
        a = r"Let $g$ be a regular morphism of schemes. Then F(g) preserves products and exact sequences."
        b = r"Let $g$ be a regular morphism of schemes and assume the image of $g$ is a dense subset of $N$. Then F(g) preserves products and exact sequences."
        self.assertIn("dense_image", [p["property"] for p in self.rewrite(a, b)["new_premises"]])
        b = a + r" The image of $g$ is a dense subset of $N$."
        self.assertNotIn("dense_image", [p["property"] for p in self.rewrite(a, b)["new_premises"]])

    def test_compound_finite_term_is_not_a_new_finiteness_premise(self):
        a = r"Let $F$ be a functor. Then F preserves products and exact sequences."
        b = r"Let $F$ be a finite-limit-preserving functor. Then F preserves products and exact sequences."
        self.assertNotIn("finite", [p["property"] for p in self.rewrite(a, b)["new_premises"]])

    def test_checked_material_nonmatch_routes_to_primary_without_certifying_silence(self):
        r = self.analyze(self.qualified_pair())
        a = r["algorithm"]
        self.assertTrue(a["new_premises"])
        self.assertEqual(a["disclosure_evidence"]["status"], "not_found_in_checked_material")
        self.assertEqual(a["investigation_queue"]["status"], "primary_unexplained_tightening")
        self.assertTrue(a["eligible_for_primary_investigation_queue"])
        self.assertEqual(a["disclosure_evidence"]["silent_change"], "unknown")
        self.assertFalse(a["disclosure_evidence"]["coverage"]["all_channel_silence_certified"])
        self.assertFalse(a["author_relation"]["historical_version_author_lists_verified"])

    def test_material_missing_and_only_context_nonmatch_are_separate(self):
        for docs in ([], [document("The theorem and its proof.", source_type="context_window")]):
            p = pair()
            p["evidence_documents"] = docs
            r = self.analyze(p)["algorithm"]
            self.assertEqual(r["disclosure_evidence"]["status"], "material_missing")
            self.assertEqual(r["investigation_queue"]["status"], "disclosure_material_missing")
            self.assertFalse(r["eligible_for_primary_investigation_queue"])

    def test_other_theorem_correction_does_not_explain_this_statement(self):
        p = self.qualified_pair()
        p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
        p["evidence_documents"].append(document("Theorem 8.4 was incorrect and has been replaced by a weaker result.", source_type="abstract"))
        r = self.analyze(p)["algorithm"]
        self.assertEqual(r["disclosure_evidence"]["status"], "paper_correction_scope_unconfirmed")
        self.assertFalse(r["disclosure_evidence"]["explains_added_premise"])
        self.assertEqual(r["investigation_queue"]["status"], "paper_correction_unconfirmed")

    def test_statement_reference_notice_does_not_certify_adequate_strength_explanation(self):
        p = self.qualified_pair()
        p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
        p["evidence_documents"].append(document("Theorem 3.2 has been revised.", source_type="version_comment"))
        r = self.analyze(p)
        d = r["algorithm"]["disclosure_evidence"]
        self.assertEqual(d["status"], "statement_explanation_found")
        self.assertFalse(d["explains_added_premise"])
        self.assertEqual(d["explanation_sufficiency"], "unknown")
        self.assertEqual(r["new"]["statement_reference_keys"], ["Theorem 3.2"])
        self.assertEqual(r["algorithm"]["investigation_queue"]["status"], "statement_notice_strength_unconfirmed")

    def test_proof_adjusted_notice_keeps_strength_scope_unknown(self):
        p = self.qualified_pair()
        p["new"]["statement_reference_keys"] = ["Lemma 4.1"]
        p["evidence_documents"].append(document("Lemma 4.7 was incorrect and removed, and the proof of Lemma 4.1 has been adjusted.", source_type="version_comment"))
        d = self.analyze(p)["algorithm"]["disclosure_evidence"]
        self.assertEqual(d["status"], "statement_explanation_found")
        self.assertIn("proof_only_strength_unconfirmed", d["statement_notice_strength_scopes"])
        self.assertFalse(d["explains_added_premise"])

    def test_explicit_missing_property_phrase_can_link_unnumbered_version_comment(self):
        a = r"Let X be a compact metric space. The orbit through $\lambda$ has dimension at least $F(X,\lambda)=2r+1$."
        b = r"Let X be a compact metric space. The orbit through $\lambda$ lying on some rational line in $V^*$ has dimension at least $F(X,\lambda)=2r+1$."
        p = pair(side("paper", a, version=1), side("paper", b, version=2))
        p["evidence_documents"] = [document("A mistake, the missing assumption that lambda is on a rational line, is now corrected.", source_type="version_comment")]
        d = self.analyze(p)["algorithm"]["disclosure_evidence"]
        self.assertEqual(d["status"], "statement_explanation_found")
        self.assertTrue(d["explains_added_premise"])
        self.assertEqual(d["matched_evidence"][0]["matched_new_property_phrases"], ["rational line"])

    def test_same_property_in_explicit_other_theorem_notice_is_not_linked(self):
        p = self.qualified_pair()
        p["new"]["statement_reference_keys"] = ["Lemma 4.1"]
        p["evidence_documents"].append(document("Theorem 8.4 was corrected by adding a finite field extension assumption.", source_type="version_comment"))
        d = self.analyze(p)["algorithm"]["disclosure_evidence"]
        self.assertEqual(d["status"], "paper_correction_scope_unconfirmed")
        self.assertFalse(d["explains_added_premise"])

    def test_old_version_correction_is_background_not_new_explanation(self):
        p = self.qualified_pair()
        p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
        old_doc = document("Theorem 3.2 was incorrect and is now corrected.", version=1)
        old_doc["side"] = "old"
        p["evidence_documents"].append(old_doc)
        d = self.analyze(p)["algorithm"]["disclosure_evidence"]
        self.assertEqual(d["status"], "not_found_in_checked_material")
        self.assertTrue(any(m["historical_background_only"] for m in d["matched_evidence"]))

    def test_other_version_unrelated_paper_and_unavailable_docs_do_not_cover_current(self):
        documents = [document("A correction of Theorem 3.2.", version=9),
                     document("A correction of Theorem 3.2.", paper_id="unrelated")]
        missing = document("A correction of Theorem 3.2.")
        missing["source_available"] = False
        documents.append(missing)
        for doc in documents:
            p = pair()
            p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
            p["evidence_documents"] = [doc]
            with self.subTest(document=doc):
                self.assertEqual(self.analyze(p)["algorithm"]["disclosure_evidence"]["status"], "material_missing")

    def test_unknown_snapshot_cannot_claim_version_comment_covers_this_snapshot(self):
        p = pair(side("older", pair()["old"]["text"], date="2020-01-01", date_kind="first_submission"),
                 side("paper", pair()["new"]["text"], date="2021-01-01", date_kind="first_submission"))
        p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
        p["evidence_documents"] = [document("Theorem 3.2 was corrected.", source_type="version_comment")]
        d = self.analyze(p)["algorithm"]["disclosure_evidence"]
        self.assertEqual(d["status"], "paper_correction_scope_unconfirmed")
        self.assertEqual(d["matched_evidence"][0]["target_version_relation"], "one_or_both_versions_unknown")

    def test_same_container_hash_for_different_text_does_not_reuse_scan(self):
        p = self.qualified_pair()
        p["evidence_documents"].append(document("A missing assumption of finite field extension is now corrected.", source_type="version_comment"))
        d = self.analyze(p)["algorithm"]["disclosure_evidence"]
        self.assertEqual(d["status"], "statement_explanation_found")

    def test_full_text_output_keeps_pointer_and_source_access_without_repeating_tex(self):
        p = self.qualified_pair()
        p["new"]["source"]["source_access"] = {"path": "not-opened.tex", "statement_start_line": 41, "statement_end_line": 49}
        r = self.analyze(p)
        self.assertEqual(r["evidence_documents"][0]["text"], "")
        self.assertEqual(r["evidence_documents"][0]["source_path"], "supplied-not-opened.tex")
        self.assertEqual(r["new"]["source"]["source_access"]["statement_start_line"], 41)
        self.assertFalse(r["algorithm"]["disclosure_evidence"]["coverage"]["external_sources_opened_by_engine"])

    def test_cached_full_frame_blocks_subset_uniqueness_upgrade(self):
        p = self.qualified_pair()
        p["cached_alignment_competition"] = {"competition": {
            "forward": {"pair_is_best": False, "margin_over_best_alternative": -.15,
                "best_alternative": {"pair_id": "closer_pair", "statement_id": "closer_statement"}},
            "reverse": {"pair_is_best": True, "margin_over_best_alternative": .6},
            "reciprocal_best_in_input": False, "near_tie_or_better_alternative": True},
            "statement_alignment": "not_best_statement_candidate", "code_version": "0.2.2",
            "frame_hash": "automatically_provided_prior_frame", "frame_name": "full_original_frame"}
        a = self.analyze(p)["algorithm"]
        self.assertEqual(a["statement_alignment"], "not_best_statement_candidate")
        self.assertFalse(a["eligible_for_primary_investigation_queue"])
        self.assertTrue(a["alignment_evidence"]["current_input_competition"]["reciprocal_best_in_input"])
        self.assertFalse(a["alignment_evidence"]["competition"]["reciprocal_best_in_input"])
        self.assertEqual(a["alignment_evidence"]["competition"]["forward"]["best_alternative"]["pair_id"], "closer_pair")

    def test_without_cached_frame_uniqueness_scope_is_explicitly_current_input_only(self):
        a = self.analyze(self.qualified_pair())["algorithm"]
        self.assertFalse(a["alignment_evidence"]["competition_provenance"]["subset_uniqueness_is_global_uniqueness"])

    def test_official_utc_version_dates_are_recognized_and_conflicts_still_block(self):
        p = pair()
        p["old"].update(date="2020-01-01T12:00:00+00:00", date_kind="official_arxiv_version_history_utc")
        p["new"].update(date="2021-01-01T12:00:00+00:00", date_kind="official_arxiv_version_history_utc")
        self.assertTrue(self.analyze(p)["algorithm"]["chronology"]["dates_are_statement_snapshot_dates"])
        p["new"]["date"] = "2019-01-01T12:00:00+00:00"
        self.assertEqual(self.analyze(p)["algorithm"]["chronology"]["status"], "conflict")

    def test_third_party_new_premise_is_retained_in_auxiliary_route(self):
        p = self.qualified_pair()
        p["old"].update(paper_id="older", authors=["Original"], version=None, date="2020-01-01", date_kind="first_submission")
        p["new"].update(authors=["Another"], version=None, date="2021-01-01", date_kind="first_submission")
        a = self.analyze(p)["algorithm"]
        self.assertEqual(a["investigation_queue"]["status"], "third_party_auxiliary")
        self.assertTrue(a["new_premises"])
        self.assertFalse(a["eligible_for_primary_investigation_queue"])

    def test_foreign_citation_number_cannot_link_current_same_number(self):
        for notice in (r"Theorem 3.2 of \cite{ExternalPaper} is incorrect.",
                       r"\cite[Theorem 3.2]{ExternalPaper} is incorrect.",
                       r"Theorem 3.2 in \cite{ExternalPaper} was corrected by adding a finite field extension assumption."):
            p = self.qualified_pair()
            p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
            p["evidence_documents"].append(document(notice))
            with self.subTest(notice=notice):
                d = self.analyze(p)["algorithm"]["disclosure_evidence"]
                self.assertEqual(d["status"], "paper_correction_scope_unconfirmed")
                self.assertFalse(d["explains_added_premise"])
                self.assertTrue(any(m["external_citation_reference_keys_excluded"] for m in d["matched_evidence"]))

    def test_first_active_end_document_excludes_appended_correction(self):
        p = self.qualified_pair()
        p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
        p["evidence_documents"] = [document(r"\begin{document}The construction is presented.\end{document}" + "\n" + "Theorem 3.2 was incorrect and is now corrected.")]
        d = self.analyze(p)["algorithm"]["disclosure_evidence"]
        self.assertEqual(d["status"], "not_found_in_checked_material")
        self.assertTrue(d["coverage"]["documents"][0]["bounded_to_first_active_end_document"])
        self.assertGreater(d["coverage"]["documents"][0]["post_document_characters_excluded"], 0)

    def test_commented_end_document_does_not_hide_active_revision_notice(self):
        p = self.qualified_pair()
        p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
        p["evidence_documents"] = [document("% \\end{document}\nTheorem 3.2 was corrected by adding the finite field extension assumption.\\end{document}")]
        a = self.analyze(p)["algorithm"]
        self.assertEqual(a["disclosure_evidence"]["status"], "statement_explanation_found")
        self.assertTrue(a["disclosure_evidence"]["explains_added_premise"])
        self.assertEqual(a["investigation_queue"]["status"], "explained_change_control")

    def test_literal_body_reference_to_other_lemma_is_not_own_statement_number(self):
        p = self.qualified_pair()
        p["new"]["text"] += " The notation of Lemma 9.3 is used here."
        p["evidence_documents"].append(document("Lemma 9.3 was corrected.", source_type="version_comment"))
        self.assertEqual(self.analyze(p)["algorithm"]["disclosure_evidence"]["status"], "paper_correction_scope_unconfirmed")

    def test_disclosure_material_limits_and_quote_coordinates_are_explicit(self):
        p = self.qualified_pair()
        p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
        raw = "Theorem 3.2 has been corrected. " + "x" * 1000100
        p["evidence_documents"] = [document(raw)]
        d = self.analyze(p)["algorithm"]["disclosure_evidence"]
        self.assertTrue(d["coverage"]["documents"][0]["scan_truncated"])
        self.assertEqual(d["coverage"]["documents"][0]["scanned_characters"], 1000000)
        item = d["matched_evidence"][0]
        start, end = item["span"]
        self.assertEqual(raw[start:end][:800], item["quote"])
        self.assertEqual(item["source_coordinate_conversion"], "not_performed")

    def test_error_correction_and_mathematical_replacement_are_not_revision_notices(self):
        for text in ("The error correction property defines the code. We design error-correction codes.",
                     "The correction factor and correction terms are bounded by the formula.",
                     "The structure group is replaced by a groupoid in the construction.",
                     "Theorem 3.2 gives a weaker estimate than the general theorem."):
            p = self.qualified_pair()
            p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
            p["evidence_documents"] = [document(text)]
            with self.subTest(text=text):
                a = self.analyze(p)["algorithm"]
                self.assertEqual(a["disclosure_evidence"]["status"], "not_found_in_checked_material")
                self.assertEqual(a["investigation_queue"]["status"], "primary_unexplained_tightening")

    def test_genuine_revision_cue_not_lost_after_incidental_same_paragraph_keyword(self):
        p = self.qualified_pair()
        p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
        p["evidence_documents"] = [document("The error correction property is used in coding; a mistake in Theorem 3.2 is now corrected by adding the finite field extension assumption.")]
        d = self.analyze(p)["algorithm"]["disclosure_evidence"]
        self.assertEqual(d["status"], "statement_explanation_found")
        self.assertTrue(d["explains_added_premise"])

    def test_independent_erratum_own_version_is_not_target_paper_version(self):
        p = self.qualified_pair()
        p["new"]["statement_reference_keys"] = ["Theorem 3.2"]
        p["evidence_documents"] = [document("Theorem 3.2 was corrected by adding a finite field extension assumption.",
            source_type="public_erratum", version=None, source_paper_id="separate-erratum", source_version=1)]
        r = self.analyze(p)
        d = r["algorithm"]["disclosure_evidence"]
        self.assertEqual(d["status"], "statement_explanation_found")
        self.assertTrue(d["explains_added_premise"])
        self.assertEqual(r["evidence_documents"][0]["source_version"], 1)
        self.assertIsNone(r["evidence_documents"][0]["version"])
        self.assertEqual(r["new"]["version"], 2)


class TestAlgorithm031(unittest.TestCase):
    def notice(self, text, source_type="full_tex", reference="Theorem 1.1", **metadata):
        p = pair()
        p["new"]["statement_reference_keys"] = [reference]
        p["evidence_documents"] = [document(text, source_type=source_type, **metadata)]
        return analyze_pairs([p])[0]["algorithm"]

    def assert_ordinary_technical_text(self, text):
        a = self.notice(text)
        d = a["disclosure_evidence"]
        self.assertEqual(d["status"], "not_found_in_checked_material")
        self.assertFalse(d["explains_added_premise"])
        self.assertFalse(any(m["statement_association_established_by_text_rule"] for m in d["matched_evidence"]))
        self.assertEqual(a["investigation_queue"]["status"], "primary_unexplained_tightening")
        self.assertEqual(d["silent_change"], "unknown")

    def test_numerical_and_estimation_corrections_are_not_publication_revisions(self):
        examples = (
            "We will find a correction to the outer expansion; the higher order terms are removed.",
            "We can expand the error in powers of the small parameter.",
            "We state the nonlinear problem for the correction.",
            "We reduce the model error in developing accurate surrogate models.",
            "Overestimating gamma changes the tail; this is corrected with the observance of extreme events.",
            "An incorrect estimation changes the numerical state, and the algorithm corrects the sign.",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assert_ordinary_technical_text(text)

    def test_current_theorem_reference_near_numerical_error_does_not_link_notice(self):
        for text in (
            "Theorem 1.1 is applied when we can expand the error in powers of the small parameter.",
            "In applying Theorem 1.1 we reduce the model error in developing a finite field extension approximation.",
            "Theorem 1.1 gives the nonlinear problem for the correction to the outer expansion.",
            "According to Theorem 1.1 an incorrect estimation occurs, and the algorithm corrects the sign.",
            "Using Theorem 1.1 we may overestimate gamma; this is corrected with the observance of extreme events.",
        ):
            with self.subTest(text=text):
                self.assert_ordinary_technical_text(text)

    def test_corrected_quantity_with_theorem_inside_prepositional_phrase_is_not_claim_subject(self):
        for adjective in ("model", "approximation", "estimation", "numerical"):
            text = "The " + adjective + " error in Theorem 1.1 is now corrected by observing the extreme events."
            with self.subTest(adjective=adjective):
                self.assert_ordinary_technical_text(text)

    def test_theorems_possessive_error_is_not_the_complete_corrected_object(self):
        for text in (
            "The estimator corrects Theorem 1.1's model error on each iteration.",
            "We find a correction to Theorem 1.1's estimation error.",
            "We corrected the error in Theorem 1.1's numerical approximation at each time step.",
        ):
            with self.subTest(text=text):
                self.assert_ordinary_technical_text(text)

    def test_new_assumption_in_current_statement_does_not_explain_its_own_addition(self):
        for text in (
            "Theorem 1.1 holds under the additional assumption of finite field extension.",
            r"\begin{theorem}Under the additional assumption of finite field extension, the construction is defined.\end{theorem}",
            "To prove Theorem 1.1 we work under the additional assumption that the field extension is finite.",
        ):
            with self.subTest(text=text):
                self.assert_ordinary_technical_text(text)

    def test_algorithm_version_is_not_earlier_publication_version(self):
        for text in (
            "The previous version of the algorithm uses a correction to the outer expansion.",
            "The previous version of the numerical estimator was corrected by observing the extreme events.",
            "The earlier version of the model is corrected by evaluating Theorem 1.1 at each step.",
        ):
            with self.subTest(text=text):
                self.assert_ordinary_technical_text(text)

    def test_previous_publication_near_quantity_correction_is_not_enough(self):
        for text in (
            "In the previous version of this paper we expand the error in powers of the parameter.",
            "In the earlier paper the model error is corrected by the numerical estimator.",
            "Theorem 1.1 of the previous paper estimates the error in developing a surrogate model.",
        ):
            with self.subTest(text=text):
                self.assert_ordinary_technical_text(text)

    def test_direct_claim_revision_predicates_still_associate_statement(self):
        for text in (
            "Theorem 1.1 was incorrect and has been withdrawn.",
            "Theorem 1.1 has been corrected.",
            "The statement of Theorem 1.1 has been revised.",
            "We corrected Theorem 1.1.",
            "A correction to Theorem 1.1 is given here.",
        ):
            with self.subTest(text=text):
                a = self.notice(text)
                d = a["disclosure_evidence"]
                self.assertEqual(d["status"], "statement_explanation_found")
                self.assertFalse(d["explains_added_premise"])
                self.assertEqual(a["investigation_queue"]["status"], "statement_notice_strength_unconfirmed")
                self.assertEqual(d["explanation_sufficiency"], "unknown")

    def test_full_tex_proof_adjustment_is_preserved_as_strength_unconfirmed(self):
        for text in (
            "The proof of Theorem 1.1 has been adjusted.",
            "We corrected the proof of Theorem 1.1.",
            "We found an error in the proof of Theorem 1.1.",
        ):
            with self.subTest(text=text):
                a = self.notice(text)
                d = a["disclosure_evidence"]
                self.assertEqual(d["status"], "statement_explanation_found")
                self.assertFalse(d["explains_added_premise"])
                self.assertIn("proof_only_strength_unconfirmed", d["statement_notice_strength_scopes"])

    def test_explicit_error_finding_in_prior_publication_remains_paper_level_notice(self):
        for text in (
            "We thank the referee for pointing out an error in a previous version of the paper.",
            "We corrected an error in our earlier paper.",
            "An error in the previous version of this paper has now been corrected.",
        ):
            with self.subTest(text=text):
                d = self.notice(text)["disclosure_evidence"]
                self.assertEqual(d["status"], "paper_correction_scope_unconfirmed")
                self.assertFalse(d["explains_added_premise"])

    def test_missing_assumption_with_actual_restoration_is_preserved(self):
        for text in (
            "The missing assumption of finite field extension has now been restored.",
            "We added the missing assumption of finite field extension.",
            "We corrected the omitted assumption of finite field extension.",
        ):
            with self.subTest(text=text):
                d = self.notice(text)["disclosure_evidence"]
                self.assertEqual(d["status"], "statement_explanation_found")
                self.assertTrue(d["explains_added_premise"])

    def test_additional_assumption_with_prior_publication_and_edit_predicate_is_notice(self):
        text = "Compared with the previous version of this paper, we added an additional assumption of finite field extension."
        d = self.notice(text)["disclosure_evidence"]
        self.assertEqual(d["status"], "statement_explanation_found")
        self.assertTrue(d["explains_added_premise"])

    def test_missing_property_in_version_comment_and_independent_erratum_are_preserved(self):
        text = "A missing assumption of finite field extension is now corrected."
        for source_type in ("version_comment", "public_erratum"):
            with self.subTest(source_type=source_type):
                a = self.notice(text, source_type=source_type)
                self.assertEqual(a["disclosure_evidence"]["status"], "statement_explanation_found")
                self.assertTrue(a["disclosure_evidence"]["explains_added_premise"])
                self.assertEqual(a["decision_status"]["mathematical_weakening"], "unknown")

    def test_true_foreign_claim_correction_stays_unlinked_to_same_current_number(self):
        for text in (r"Theorem 1.1 of \cite{ExternalWork} was incorrect.",
                     r"\cite[Theorem 1.1]{ExternalWork} has been corrected."):
            with self.subTest(text=text):
                d = self.notice(text)["disclosure_evidence"]
                self.assertEqual(d["status"], "paper_correction_scope_unconfirmed")
                self.assertFalse(d["explains_added_premise"])


if __name__ == "__main__":
    unittest.main()
