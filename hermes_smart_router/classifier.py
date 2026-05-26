"""Complexity classifier for Hermes Smart Router.

Two classification strategies:
1. Regex-based (fast, free, default) — uses pattern matching and heuristics
2. LLM-based (optional, configurable) — calls a cheap LLM for uncertain cases
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger(__name__)

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


def llm_classify_message(
    text: str,
    provider: str = "nous",
    model: str = "deepseek-v4-free",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Classification:
    """Classify using an LLM, falling back to regex on any failure.

    The LLM is called with a simple system prompt asking it to classify
    the message as simple/medium/complex and return JSON. On failure
    (network error, timeout, bad response), we silently fall back to
    the regex-based classify_message().

    Environment variables for API keys are resolved automatically:
    - ``NOUS_API_KEY`` for nous provider
    - ``OPENAI_API_KEY`` for openai-codex provider
    - ``DEEPSEEK_API_KEY`` for deepseek/openai-compatible providers
    """
    try:
        classification = _try_llm_classify(text, provider, model, api_key, base_url)
        if classification is not None:
            return classification
    except Exception:
        logger.debug("LLM classifier failed, falling back to regex", exc_info=True)

    # Fall back to regex-based classification
    return classify_message(text)


def _resolve_api_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    """Resolve API key from param or environment."""
    if api_key:
        return api_key

    env_map = {
        "nous": "NOUS_API_KEY",
        "openai-codex": "OPENAI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "opencode-go": "DEEPSEEK_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    env_var = env_map.get(provider)
    if env_var:
        return os.environ.get(env_var)

    # Generic fallback: try common env vars
    for var in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "NOUS_API_KEY"):
        key = os.environ.get(var)
        if key:
            return key

    return None


def _resolve_base_url(provider: str, base_url: Optional[str] = None) -> Optional[str]:
    """Resolve base URL from param or known provider defaults."""
    if base_url:
        return base_url

    defaults = {
        "nous": "https://api.nousresearch.com/v1",
        "openai-codex": "https://api.openai.com/v1",
        "openai": "https://api.openai.com/v1",
        "opencode-go": "https://api.deepseek.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }
    return defaults.get(provider)


def _try_llm_classify(
    text: str,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[Classification]:
    """Attempt LLM classification. Returns None if anything fails."""
    import urllib.request

    key = _resolve_api_key(provider, api_key)
    if not key:
        logger.debug("No API key available for LLM classifier (provider=%s)", provider)
        return None

    url = _resolve_base_url(provider, base_url)
    if not url:
        logger.debug("No base URL for provider %s", provider)
        return None

    endpoint = f"{url.rstrip('/')}/chat/completions"

    system_prompt = (
        "You are a task complexity classifier. Classify the user message as one of:\n"
        '- "simple": greetings, thanks, yes/no, basic facts, simple lookups\n'
        '- "medium": explanations, comparisons, planning, analysis, drafting\n'
        '- "complex": implementation, debugging, refactoring, testing, deployment, architecture\n'
        "Respond with ONLY valid JSON: {\"complexity\": \"...\", \"reason\": \"...\"}\n"
        "No other text."
    )

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text[:2000]},  # Truncate to avoid token waste
        ],
        "temperature": 0,
        "max_tokens": 100,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
    except Exception as e:
        logger.debug("LLM classifier HTTP request failed: %s", e)
        return None

    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        result = json.loads(content)
        complexity = result.get("complexity", "").lower()
        reason = result.get("reason", "LLM classified")
        if complexity in ("simple", "medium", "complex"):
            return Classification(complexity, 99, reason)  # Score 99 = LLM-classified
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        logger.debug("LLM classifier response parsing failed: %s", e)
        return None

    return None
