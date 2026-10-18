# Working rules

- Read README.md and REVIEW_GUIDE.md before changing the code.
- The runtime is deterministic and consumes no per-pair human or model verdicts.
- Source handling is original-corpus-first, read-only, bounded, and version-conscious. Statement overlap never proves version identity.
- Keep old scan evidence hashes and byte recovery when changing source navigation or removing redundant files.
- Use synthetic regression cases; do not access a full external corpus or download papers for ordinary tests.
- Fix general rules, not selected paper IDs. Do not claim true mathematical recall from candidate counts.
- Run `python -X utf8 -m unittest discover -v` for relevant changes.
