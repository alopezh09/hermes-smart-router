"""Shared provider defaults for Hermes Smart Router.

Centralised source of truth for API keys, base URLs, and API modes
used by both the classifier LLM calls and the gateway session routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Per-provider runtime defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderDefaults:
    env_key: str           # e.g. "NOUS_API_KEY"
    base_url: str          # e.g. "https://inference-api.nousresearch.com/v1"
    api_mode: str = "chat_completions"


PROVIDERS: Dict[str, ProviderDefaults] = {
    "nous": ProviderDefaults(
        env_key="NOUS_API_KEY",
        base_url="https://inference-api.nousresearch.com/v1",
    ),
    "openai-codex": ProviderDefaults(
        env_key="OPENAI_API_KEY",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
    ),
    "openai": ProviderDefaults(
        env_key="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
    ),
    "opencode-go": ProviderDefaults(
        env_key="OPENCODE_GO_API_KEY",
        base_url="https://opencode.ai/zen/go/v1",
    ),
    "deepseek": ProviderDefaults(
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
    ),
}


def resolve_api_key(provider: str, explicit: Optional[str] = None) -> Optional[str]:
    """Return an API key for *provider*, preferring *explicit*."""
    import os

    if explicit:
        return explicit
    info = PROVIDERS.get(provider)
    if info and info.env_key:
        return os.environ.get(info.env_key)
    # Generic fallback — try common env vars
    for var in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "NOUS_API_KEY", "OPENCODE_GO_API_KEY"):
        key = os.environ.get(var)
        if key:
            return key
    return None


def resolve_base_url(provider: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    info = PROVIDERS.get(provider)
    return info.base_url if info else ""


def resolve_api_mode(provider: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    info = PROVIDERS.get(provider)
    return info.api_mode if info else "chat_completions"
