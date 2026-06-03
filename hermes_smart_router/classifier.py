"""LLM-based complexity classifier for Hermes Smart Router.

Classification is always done via an LLM call. No regex fallback.
If the LLM call fails (network error, no API key, bad response),
classification fails and the router lets Hermes use its default model.

The LLM provider and model are configurable via the smart_router section
in config.yaml.  Authentication is resolved through Hermes' own provider
system (``resolve_runtime_provider``), not separate API keys — the same
mechanism Hermes uses for all its providers.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger(__name__)

Complexity = Literal["simple", "medium", "complex"]


@dataclass(frozen=True)
class Classification:
    complexity: Complexity
    score: int
    reason: str


# ---------------------------------------------------------------------------
# LLM-based classifier (always active — no regex fallback)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a task complexity router. Classify the user message into one of three tiers.\n"
    "Return ONLY valid JSON: {\"complexity\": \"simple|medium|complex\", \"score\": 0-10, \"reason\": \"brief explanation\"}\n\n"
    "TIER DEFINITIONS:\n"
    "- simple (score 0-1): Greetings, thanks, yes/no, basic facts, trivial lookups, chitchat.\n"
    "- medium (score 2-5): Explanations, comparisons, planning, analysis, drafting, research, documentation.\n"
    "- complex (score 6-10): Implementation, coding, debugging, refactoring, testing, deployment, architecture,\n"
    "  building apps/systems/APIs, database design, security, infrastructure, DevOps, ML training,\n"
    "  multi-step tasks, file manipulation, system configuration, full-stack development.\n\n"
    "KEY INDICATORS OF COMPLEX (score 6+):\n"
    "- Mentions of: app, application, backend, frontend, API, database, server, deploy, build, create, develop\n"
    "- Building/creating something from scratch\n"
    "- Multiple technologies or components mentioned together\n"
    "- Financial/banking/transaction systems\n"
    "- Code generation, refactoring, debugging\n"
    "- End-to-end systems, full-stack projects\n\n"
    "Be conservative: if unsure, prefer medium over simple, and complex over medium."
)


def classify_message(
    text: str,
    provider: str = "nous",
    model: str = "openrouter/owl-alpha",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[Classification]:
    """Classify an incoming message by calling an LLM.

    Parameters
    ----------
    text:
        The incoming message text.
    provider:
        Provider name for API key / base URL resolution.
    model:
        Model name to use for classification.
    api_key:
        Optional explicit API key. Falls back to environment variables.
    base_url:
        Optional explicit base URL. Falls back to known provider defaults.

    Returns
    -------
    Classification or None
        Returns ``None`` if the LLM call fails — the caller should
        treat this as a signal to skip routing (fail open).
    """

    if not text or not text.strip():
        return None

    try:
        return _try_llm_classify(text, provider, model, api_key, base_url)
    except Exception:
        logger.debug("LLM classifier failed", exc_info=True)
        return None


def _resolve_api_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    """Resolve API key from Hermes runtime provider, then env vars, then param."""
    if api_key:
        return api_key

    # 1. Use Hermes' own provider resolution (JWT, token pools, etc.)
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        resolved = resolve_runtime_provider(requested=provider)
        key = resolved.get("api_key") if isinstance(resolved, dict) else None
        if key:
            return key
    except Exception:
        logger.debug("resolve_runtime_provider failed for %s, trying env vars", provider, exc_info=True)

    # 2. Fall back to environment variables
    env_map = {
        "nous": "NOUS_API_KEY",
        "openai-codex": "OPENAI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "opencode-go": "OPENCODE_GO_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    env_var = env_map.get(provider)
    if env_var:
        return os.environ.get(env_var)

    # 3. Generic fallback
    for var in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "NOUS_API_KEY"):
        key = os.environ.get(var)
        if key:
            return key

    return None


def _resolve_base_url(provider: str, base_url: Optional[str] = None) -> Optional[str]:
    """Resolve base URL from Hermes runtime provider, then param, then defaults."""
    if base_url:
        return base_url

    # 1. Use Hermes' own provider resolution
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        resolved = resolve_runtime_provider(requested=provider)
        url = resolved.get("base_url") if isinstance(resolved, dict) else None
        if url:
            return url
    except Exception:
        logger.debug("resolve_runtime_provider base_url failed for %s", provider, exc_info=True)

    # 2. Fall back to known defaults
    defaults = {
        "nous": "https://inference-api.nousresearch.com/v1",
        "openai-codex": "https://api.openai.com/v1",
        "openai": "https://api.openai.com/v1",
        "opencode-go": "https://api.deepseek.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }
    return defaults.get(provider)


def _parse_json_response(content: str) -> dict:
    """Parse JSON from an LLM response, handling markdown code fences.

    Many LLMs wrap JSON in ```json ... ``` blocks. This strips those
    before attempting to parse.
    """
    text = content.strip()
    if text.startswith("```"):
        # Strip opening fence: ```json or ```
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Strip closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return json.loads(text)


def _try_llm_classify(
    text: str,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[Classification]:
    """Attempt LLM classification. Returns None if anything fails."""

    key = _resolve_api_key(provider, api_key)
    if not key:
        logger.debug("No API key available for LLM classifier (provider=%s)", provider)
        return None

    url = _resolve_base_url(provider, base_url)
    if not url:
        logger.debug("No base URL for provider %s", provider)
        return None

    endpoint = f"{url.rstrip('/')}/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
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
        result = _parse_json_response(content)
        complexity = result.get("complexity", "").lower()
        reason = result.get("reason", "LLM classified")
        llm_score = result.get("score", None)
        if complexity in ("simple", "medium", "complex"):
            # Use LLM's own score if provided, otherwise map label to score
            if isinstance(llm_score, (int, float)) and 0 <= llm_score <= 10:
                score = int(llm_score)
            elif complexity == "complex":
                score = 8
            elif complexity == "medium":
                score = 4
            else:
                score = 1
            return Classification(complexity, score, reason)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        logger.debug("LLM classifier response parsing failed: %s", e)
        return None

    return None
