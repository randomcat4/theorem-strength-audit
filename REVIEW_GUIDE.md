# Review brief for other models

This repository is a deterministic screening algorithm for possible changes in mathematical claim strength. It consumes already retrieved candidate pairs. It is not a theorem prover, a full-corpus retriever, or a classifier of misconduct.

Please inspect the code and run the synthetic tests. Report concrete failure cases, minimal reproductions and the affected lines. Do not manufacture a verdict on real papers or infer recall from the selected candidate set.

## Module map

| Module | Responsibility |
|---|---|
| `strength_signals.py` | Entry point `analyze_pairs`: temporal/author metadata, change signals, scoring, gates and reading queues |
| `statement_text.py` | Shared primitives: active-text masking, tokens, similarity, formula units, signal records |
| `alignment.py` | Statement alignment evidence and in-batch competition |
| `premises.py` | Added premises on the same textual object: declared properties, modifiers, hypothesis clauses |
| `disclosure.py` | Revision notices, errata, full-TeX scans and their association with the statement |
| `formula_scope.py` | Bounded formula atoms, clause roles, local range inclusion, unsupported syntax and edit ledger |
| `asymptotic_order.py` | Single-variable power/log order comparison of paired O/o/Ω/ω/Θ bounds in the conclusion |
| `static_tex_context.py` | Restricted source-backed literal aliases and identity wrappers |
| `render_results.py` | Human and compact machine-readable evidence, with uncertainty and source navigation |
| `run_algorithm.py` | Bounded JSONL execution and input/code/output hashes |
| `tex_sources.py` | Original-corpus access, strict version bindings, lossless old evidence access and localhost reader |
| `prepare_recheck99.py` | Explicit-input source/evidence adapter; no verdict labels |
| `tools/migrate_tex_sources.py` | Operational migration plan, report backups, link rewriting, verification and allow-listed cleanup |

## Highest-value questions

1. Can new assumptions, conclusion restrictions, existential choices, branch conditions and parameter declarations be confused?
2. Can an object or constraint in one enumeration item leak into another item?
3. Do alignment competition and metadata uncertainty actually block misleading high-priority alerts?
4. Is a rewrite suppressed only when the complete active change is covered by supported rules?
5. Can a paper-level notice, another theorem's correction, or the current premise itself falsely explain a revision?
6. Is local interval/set containment being overstated as an overall theorem weakening?
7. Can a matching statement be mistaken for proof that a whole corpus snapshot is a particular version?
8. Do source bytes, line coordinates, retained scan evidence and cleanup recovery remain distinct and reproducible?
9. Do runtime paths read reader feedback, manual verdicts, network services or language models?
10. Where could the retrieval, extraction or alignment stage miss changes before the detector even sees them?

## Important constraints

- Keep the runtime non-LLM. Humans and models consume output; they do not supply per-pair judgments to the algorithm.
- No full-corpus rerun is required to review this code. Tests use small synthetic inputs and an optional tiny gzip fixture.
- Original corpus access is read-only and uses existing indexes. Missing or unbound versions must remain explicit.
- Preserve exact old evidence when changing navigation. The text previously scanned by a detector may differ from the canonical corpus bytes.
- Do not add paper-ID-specific exceptions or alter frozen sample membership to improve reported results.
- Tests demonstrate finite behavior, not mathematical truth or population recall.

Please distinguish confirmed implementation bugs, semantic limitations, and hypotheses requiring additional evidence. Suggested edits should include adversarial tests that exercise the actual issue.
