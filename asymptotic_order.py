"""Bounded comparison of asymptotic bounds such as O(n log n) -> O(n^2).

Only single-variable products of powers and logarithms are compared, e.g.
``n^{3/2}``, ``\\sqrt{n}\\log^2 n`` or ``(\\log n)^{2}``. Anything else is left
unparsed and produces no comparison. A result says how one located bound
moved; it never certifies that the whole statement became weaker.
"""
from __future__ import annotations

from fractions import Fraction
import re

_OPERATOR = re.compile(r"(?<![A-Za-z\\])(?P<op>O|o|\\mathcal\s*\{\s*O\s*\}|\\Omega|\\Theta|\\omega)\s*(?:\\left|\\bigl?|\\Bigl?)?\s*\(")
_KIND = {"O": "upper", "o": "upper", "Omega": "lower", "omega": "lower", "Theta": "tight"}
_NUMBER = r"\d+(?:\.\d+)?(?:/\d+)?"
_FACTORS = [
    ("sqrt", re.compile(r"\\sqrt\{(?P<var>[A-Za-z])\}")),
    ("log_power_paren", re.compile(r"\(\\log(?P<var>[A-Za-z])\)\^\{?(?P<exp>" + _NUMBER + r")\}?")),
    ("log_power", re.compile(r"\\log\^\{?(?P<exp>" + _NUMBER + r")\}?\(?(?P<var>[A-Za-z])\)?")),
    ("log", re.compile(r"\\log\(?(?P<var>[A-Za-z])\)?")),
    ("power", re.compile(r"(?P<var>[A-Za-z])\^\{?(?P<exp>" + _NUMBER + r")\}?")),
    ("frac_power", re.compile(r"(?P<var>[A-Za-z])\^\{\\frac\{(?P<num>\d+)\}\{(?P<den>\d+)\}\}")),
    ("variable", re.compile(r"(?P<var>[A-Za-z])")),
    ("constant", re.compile(_NUMBER)),
]
_CONCLUSION = re.compile(r"\bthen\b|\bwe\s+have\b|\\(?:Longrightarrow|implies)(?![A-Za-z])|⇒", re.I)
_HYPOTHESIS_CUE = re.compile(r"\b(?:let|suppose|assume|assuming|if|provided)\b", re.I)


def growth(argument: str) -> tuple[str, Fraction, Fraction] | None:
    """Return (variable, power exponent, log exponent), or None if unsupported."""
    text = re.sub(r"\\left|\\right|\\cdot|\\,|\\;|\\!|\s+|\*", "", argument)
    variable, power, logs, position = None, Fraction(0), Fraction(0), 0
    while position < len(text):
        for kind, pattern in _FACTORS:
            match = pattern.match(text, position)
            if match:
                break
        else:
            return None
        if kind != "constant":
            if variable not in (None, match.group("var")):
                return None
            variable = match.group("var")
        if kind == "sqrt":
            power += Fraction(1, 2)
        elif kind in {"log_power_paren", "log_power"}:
            logs += Fraction(match.group("exp"))
        elif kind == "log":
            logs += 1
        elif kind == "power":
            power += Fraction(match.group("exp"))
        elif kind == "frac_power":
            power += Fraction(int(match.group("num")), int(match.group("den")))
        elif kind == "variable":
            power += 1
        position = match.end()
    return (variable, power, logs) if variable else None


def _argument(text: str, start: int) -> tuple[str, int] | None:
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start:index], index + 1
    return None


def _conclusion_region(text: str) -> tuple[int, str]:
    marker = _CONCLUSION.search(text)
    if marker:
        return marker.end(), "conclusion_after_marker"
    if _HYPOTHESIS_CUE.search(text):
        return len(text), "no_conclusion_marker"  # Bounds might be hypotheses; skip.
    return 0, "whole_statement_without_hypothesis_cue"


def bounds(text: str) -> list[dict]:
    start, role = _conclusion_region(text)
    found = []
    for match in _OPERATOR.finditer(text, start):
        parsed = _argument(text, match.end())
        if not parsed:
            continue
        argument, end = parsed
        operator = re.sub(r"\\mathcal\s*\{\s*O\s*\}", "O", match.group("op")).lstrip("\\")
        left = text[max(start, match.start() - 60):match.start()]
        subject = re.split(r"[$,;.]|\\\\|\band\b", left)[-1]
        subject = re.sub(r"\s+|=|\\(?:le|leq|ge|geq|in|sim)(?![A-Za-z])|\bis\b|\bof\b", "", subject)
        found.append({"operator": operator, "kind": _KIND[operator], "subject": subject,
                      "argument": argument.strip(), "growth": growth(argument),
                      "start": match.start(), "end": end, "role": role})
    return found


def _order(value: tuple[str, Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return value[1], value[2]


def compare_bounds(old: str, new: str) -> list[dict]:
    """Pair bounds with the same subject and operator kind, one on each side."""
    olds, news = bounds(old), bounds(new)
    findings = []
    for key in {(b["subject"], b["kind"]) for b in news}:
        a = [b for b in olds if (b["subject"], b["kind"]) == key]
        b = [b for b in news if (b["subject"], b["kind"]) == key]
        if len(a) != 1 or len(b) != 1:
            continue
        a, b = a[0], b[0]
        if not a["growth"] or not b["growth"] or a["growth"][0] != b["growth"][0]:
            continue
        before, after = _order(a["growth"]), _order(b["growth"])
        if before == after:
            continue
        larger = after > before
        if b["kind"] == "tight":
            direction = "changed_tight_bound"
        elif (b["kind"] == "upper") == larger:
            direction = "weaker"
        else:
            direction = "stronger"
        findings.append({"subject": b["subject"], "kind": b["kind"], "direction": direction,
                         "old_quote": old[a["start"]:a["end"]], "new_quote": new[b["start"]:b["end"]],
                         "old_span": [a["start"], a["end"]], "new_span": [b["start"], b["end"]],
                         "old_order": [str(before[0]), str(before[1])], "new_order": [str(after[0]), str(after[1])],
                         "role": b["role"]})
    return sorted(findings, key=lambda f: f["new_span"])


def mask_bound_arguments(text: str) -> str:
    """Replace each asymptotic-bound argument with one placeholder token."""
    pieces, position = [], 0
    for match in _OPERATOR.finditer(text):
        if match.start() < position:
            continue
        parsed = _argument(text, match.end())
        if not parsed:
            continue
        pieces.append(text[position:match.end()] + "BOUND)")
        position = parsed[1]
    return "".join(pieces) + text[position:]
