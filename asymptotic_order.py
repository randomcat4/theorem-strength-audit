"""Bounded comparison of asymptotic bounds such as O(n log n) -> O(n^2).

Only single-variable products of powers and logarithms are compared, e.g.
``n^{3/2}``, ``\\sqrt{n}\\log^2 n`` or ``(\\log n)^{2}``. Anything else is left
unparsed and produces no comparison. A result says how one located bound
moved; it never certifies that the whole statement became weaker. Comparisons
require the same explicit limit (positive infinity or zero); log factors near
zero and conditional/compound conclusions are outside this finite grammar.
"""
from __future__ import annotations

from fractions import Fraction
import re
from statement_text import _active_text

_OPERATOR = re.compile(r"(?<![A-Za-z\\])(?P<op>O|o|\\mathcal\s*\{\s*O\s*\}|\\Omega|\\Theta|\\omega)\s*(?:\\left|\\bigl?|\\Bigl?)?\s*\(")
_KIND = {"O": "upper", "o": "upper", "Omega": "lower", "omega": "lower", "Theta": "tight"}
_NUMBER = r"\d{1,12}(?:\.\d{1,12})?"
_EXP = r"(?:\{(?P<exp>-?" + _NUMBER + r"(?:/" + _NUMBER + r")?)\}|(?P<digit>\d))"
_FACTORS = [
    ("sqrt", re.compile(r"\\sqrt\{(?P<var>[A-Za-z])\}")),
    ("log_power_paren", re.compile(r"\(\\log(?P<var>[A-Za-z])\)\^" + _EXP)),
    ("log_power", re.compile(r"\\log\^" + _EXP + r"(?P<var>[A-Za-z])")),
    ("log", re.compile(r"\\log(?P<var>[A-Za-z])")),
    ("power", re.compile(r"(?P<var>[A-Za-z])\^" + _EXP)),
    ("frac_power", re.compile(r"(?P<var>[A-Za-z])\^\{\\frac\{(?P<num>\d+)\}\{(?P<den>\d+)\}\}")),
    ("variable", re.compile(r"(?P<var>[A-Za-z])")),
    ("constant", re.compile(r"/?" + _NUMBER)),
]
_CONCLUSION = re.compile(r"\bthen\b|\bwe\s+have\b|\\(?:Longrightarrow|implies)(?![A-Za-z])|⇒", re.I)
_HYPOTHESIS_CUE = re.compile(r"\b(?:let|suppose|assume|assuming|if|provided)\b", re.I)


def growth(argument: str) -> tuple[str, Fraction, Fraction] | None:
    """Return (variable, power exponent, log exponent), or None if unsupported."""
    if len(argument) > 1024:
        return None
    if set(re.findall(r"\\[A-Za-z]+", argument)) - {r'\sqrt', r'\log', r'\frac', r'\left', r'\right', r'\cdot'}:
        return None  # Preserve control-word boundaries before removing spaces.
    text = re.sub(r"\\(?:left|right|cdot)(?![A-Za-z])|\\,|\\;|\\!|\s+|\*", "", argument)
    text = re.sub(r"\\log\(([A-Za-z])\)", r"\\log\1", text)
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
        try:
            exponent = match.groupdict().get('exp') or match.groupdict().get('digit')
            if exponent:
                parts = exponent.split('/')
                exponent = Fraction(parts[0]) / (Fraction(parts[1]) if len(parts) == 2 else 1)
            if kind == 'constant' and (Fraction(match.group().lstrip('/')) <= 0 or (position == 0 and match.group().startswith('/'))):
                return None
            if kind == 'frac_power':
                exponent = Fraction(int(match.group('num')), int(match.group('den')))
        except (ValueError, ZeroDivisionError):
            return None
        if kind == "sqrt":
            power += Fraction(1, 2)
        elif kind in {"log_power_paren", "log_power"}:
            logs += exponent
        elif kind == "log":
            logs += 1
        elif kind == "power":
            power += exponent
        elif kind == "frac_power":
            power += exponent
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
    # A first 'then' does not parse nested implications, enumeration branches,
    # negations or alternatives. Leave those to the generic change ledger.
    if re.search(r"\b(?:if|iff|unless|provided|assuming|suppose|assume|let|not|never|or|when|whenever|where|for|condition|necessary|sufficient|satisfying)\b|\bsuch\s+that\b|\\(?:neg|lnot|implies|Longrightarrow|iff)\b", text[start:], re.I):
        return []
    if re.search(r"\\item\b|\\begin\s*\{(?:cases|enumerate)\}", text):
        return []
    found = []
    for match in _OPERATOR.finditer(text, start):
        parsed = _argument(text, match.end())
        if not parsed:
            continue
        argument, end = parsed
        operator = re.sub(r"\\mathcal\s*\{\s*O\s*\}", "O", match.group("op")).lstrip("\\")
        left = text[start:match.start()]
        # Require an explicit simple mathematical subject and equality/membership.
        # Do not erase inequalities, or pair unrelated prose with an empty subject.
        subject_match = re.search(r"(?:\$|\\\[|\\\()\s*([A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?(?:\([A-Za-z0-9]+\))?)\s*(=|\\in)\s*$", left)
        if not subject_match:
            continue
        subject = subject_match.group(1) + subject_match.group(2)
        suffix = text[end:]
        if not re.match(r"\s*(?:\$|\\\]|\\\))", suffix):
            continue  # Not the whole RHS: e.g. O(n)+n^2 or an equality chain.
        found.append({"operator": operator, "kind": _KIND[operator], "subject": subject,
                      "argument": argument.strip(), "growth": growth(argument),
                      "start": match.start(), "end": end, "role": role})
    return found


def _order(value: tuple[str, Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return value[1], value[2]


def _limit(text: str, variable: str) -> str | None:
    """Require the same explicit limit on both sides; never assume infinity."""
    targets = re.findall(r"(?<![A-Za-z])" + re.escape(variable) + r"\s*(?:\\to(?![A-Za-z])|\\rightarrow(?![A-Za-z])|→)\s*([^\s$,.;}]+)", text)
    targets = set(targets)
    if targets == {r'\infty'} or targets == {r'+\infty'} or targets == {'∞'}:
        return 'positive_infinity'
    if targets == {'0'}:
        return 'zero'
    return None


def compare_bounds(old: str, new: str) -> list[dict]:
    """Pair bounds with the same subject, exact operator and explicit limit."""
    old, new = _active_text(old), _active_text(new)
    olds, news = bounds(old), bounds(new)
    findings = []
    for key in {(b["subject"], b["operator"]) for b in news}:
        a = [b for b in olds if (b["subject"], b["operator"]) == key]
        b = [b for b in news if (b["subject"], b["operator"]) == key]
        if len(a) != 1 or len(b) != 1:
            continue
        a, b = a[0], b[0]
        if not a["growth"] or not b["growth"] or a["growth"][0] != b["growth"][0]:
            continue
        limit = _limit(old, a['growth'][0])
        if limit is None or limit != _limit(new, b['growth'][0]):
            continue
        if limit == 'zero' and (a['growth'][2] or b['growth'][2]):
            continue  # log near zero needs sign/domain/absolute-value analysis.
        before, after = _order(a["growth"]), _order(b["growth"])
        if before == after:
            continue
        larger = after > before if limit == 'positive_infinity' else after < before
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
                         "role": b["role"], "limit": limit})
    return sorted(findings, key=lambda f: f["new_span"])
