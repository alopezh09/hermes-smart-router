"""Cheap deterministic complexity classifier for the first MVP.

The classifier intentionally starts with local heuristics so routing itself does
not require a model call. Later we can add an optional cheap-LLM classifier when
these rules are uncertain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Complexity = Literal["simple", "medium", "complex"]


@dataclass(frozen=True)
class Classification:
    complexity: Complexity
    score: int
    reason: str


COMPLEX_PATTERNS = [
    r"\bimplement(a|ar|ación|ation)?\b",
    r"\brefactor\b",
    r"\bdebug\b",
    r"\bbug\b",
    r"\btest(s|ing)?\b",
    r"\bdeploy\b",
    r"\bproduction\b",
    r"\barchitecture\b",
    r"\barquitectura\b",
    r"\bplugin\b",
    r"\bgithub\b",
    r"\bpull request\b|\bPR\b",
    r"\bcodebase\b",
    r"\brepo(sitory)?\b",
    r"\bbase de código\b",
    r"\bend[- ]?to[- ]?end\b",
    r"\bcompleto\b",
    r"\bautomatiz(a|ar|ación)\b",
]

MEDIUM_PATTERNS = [
    r"\bexplica(r|me)?\b",
    r"\bexplícame\b",
    r"\bresume(n|ir)?\b",
    r"\bcompara(r)?\b",
    r"\bplan\b",
    r"\bdiseña(r)?\b",
    r"\banaliza(r)?\b",
    r"\bescribe\b",
    r"\bdraft\b",
    r"\bmejora(r)?\b",
]

SIMPLE_PATTERNS = [
    r"^(ok|dale|gracias|perfecto|sí|si|no|listo|va|claro)[.!\s]*$",
    r"\bqué hora\b",
    r"\bcuánto es\b",
    r"\bdefine\b",
]


def _count_matches(patterns: list[str], text: str) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))


def classify_message(text: str) -> Classification:
    """Classify an incoming message as simple, medium, or complex."""
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return Classification("simple", 0, "empty message")

    words = normalized.split()
    word_count = len(words)
    has_code_block = "```" in normalized
    has_many_requirements = sum(token in normalized for token in ["\n-", " 1.", " 2.", ";", " y ", " and "])

    simple_hits = _count_matches(SIMPLE_PATTERNS, normalized)
    medium_hits = _count_matches(MEDIUM_PATTERNS, normalized)
    complex_hits = _count_matches(COMPLEX_PATTERNS, normalized)

    score = 0
    score += medium_hits * 2
    score += complex_hits * 4
    score += 5 if has_code_block else 0
    score += 3 if word_count >= 80 else 0
    score += 2 if word_count >= 35 else 0
    score += min(has_many_requirements, 3)
    score -= simple_hits * 3

    if score >= 6:
        return Classification("complex", score, "complexity score >= 6")
    if score >= 2:
        return Classification("medium", score, "complexity score >= 2")
    return Classification("simple", score, "low complexity score")
