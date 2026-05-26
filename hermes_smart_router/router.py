"""Gateway hook implementation for Hermes Smart Router."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, List

from .classifier import classify_message, llm_classify_message
from .config import load_config, Route, ScoringConfig, PatternConfig

logger = logging.getLogger(__name__)

_ALLOW = {"action": "allow"}
_SKIP = {"action": "skip"}

# Store routing decisions on the gateway object so the transform_llm_output
# hook can read them. Key: session_key, Value: routing info dict.
_ROUTING_DECISIONS_ATTR = "_smart_router_decisions"


def _get_decisions_store(gateway: Any) -> dict:
    """Get or create the routing decisions dict on the gateway object."""
    if not hasattr(gateway, _ROUTING_DECISIONS_ATTR):
        setattr(gateway, _ROUTING_DECISIONS_ATTR, {})
    return getattr(gateway, _ROUTING_DECISIONS_ATTR)


def _safe_get_session_key(gateway: Any, source: Any) -> Optional[str]:
    session_key_fn = getattr(gateway, "_session_key_for_source", None)
    if callable(session_key_fn):
        key = session_key_fn(source)
        if isinstance(key, str) and key:
            return key
    return None


def _is_manual_override(existing: Any) -> bool:
    return isinstance(existing, dict) and bool(existing) and not existing.get("_smart_router")


# ---------------------------------------------------------------------------
# Slash command handling
# ---------------------------------------------------------------------------

def _handle_slash_command(event: Any, gateway: Any) -> Optional[dict]:
    """Handle /smart-router slash commands. Returns action dict or None.

    Delegates display/formatting to :mod:`hermes_smart_router.commands`.
    Toggle commands (dry-run, footer, classifier) remain here because they
    modify runtime gateway state.
    """
    text = (event.text or "").strip()
    if not text.startswith("/smart-router"):
        return None

    source = getattr(event, "source", None)
    if source is None:
        return _SKIP

    cfg = _get_effective_config(gateway)
    parts = text.split(maxsplit=1)
    subcommand = parts[1].strip() if len(parts) > 1 else ""

    # ── Delegate to commands.py for display/report commands ──
    from .commands import handle_slash_command as _cmd_handle
    response = _cmd_handle(text, gateway, source, cfg)
    if response is not None:
        # Check if this is a toggle command that also needs side effects
        if subcommand.startswith("dry-run") and not subcommand[len("dry-run"):].strip().lower() in ("on", "true", "1", "enable", "off", "false", "0", "disable", ""):
            pass  # dry-run with a message: just display
        elif subcommand.startswith("classifier") and not subcommand[len("classifier"):].strip().lower() in ("llm", "on", "true", "1", "enable", "regex", "off", "false", "0", "disable", ""):
            pass  # classifier with a message: just display
        return _reply_or_rewrite(gateway, source, response)

    # ── Legacy / toggle commands (modify runtime state) ──
    if subcommand == "status" or subcommand == "":
        # Already handled by commands.py above — but keep fallback
        return _SKIP

    elif subcommand.startswith("dry-run"):
        arg = subcommand[len("dry-run"):].strip().lower()
        if arg in ("on", "true", "1", "enable"):
            _toggle_config(gateway, "dry_run", True)
            return _reply_or_rewrite(gateway, source, "🔬 Smart Router dry-run mode **ON** — routing decisions will be logged but not applied.")
        elif arg in ("off", "false", "0", "disable"):
            _toggle_config(gateway, "dry_run", False)
            return _reply_or_rewrite(gateway, source, "🚀 Smart Router dry-run mode **OFF** — routing will be applied normally.")
        elif arg:
            return _reply_or_rewrite(gateway, source, _format_classification_preview(arg, cfg, title="🔬 Smart Router dry-run"))
        else:
            return _reply_or_rewrite(gateway, source, f"Usage: `/smart-router dry-run on|off` or `/smart-router dry-run <message>`\nCurrent: {'on' if cfg.dry_run else 'off'}")

    elif subcommand.startswith("footer"):
        arg = subcommand[len("footer"):].strip().lower()
        if arg in ("on", "true", "1", "enable"):
            _toggle_config(gateway, "show_route_footer", True)
            return _reply_or_rewrite(gateway, source, "✅ Smart Router footer **ON** — route info will appear at the end of each response.")
        elif arg in ("off", "false", "0", "disable"):
            _toggle_config(gateway, "show_route_footer", False)
            return _reply_or_rewrite(gateway, source, "❌ Smart Router footer **OFF** — route info will be hidden.")
        else:
            return _reply_or_rewrite(gateway, source, f"Usage: `/smart-router footer on|off`\nCurrent: {'on' if cfg.show_route_footer else 'off'}")

    elif subcommand.startswith("classifier"):
        arg = subcommand[len("classifier"):].strip().lower()
        if arg in ("llm", "on", "true", "1", "enable"):
            _toggle_config(gateway, "llm_classifier_enabled", True)
            return _reply_or_rewrite(gateway, source, "🤖 Smart Router classifier: **LLM** (with regex fallback)")
        elif arg in ("regex", "off", "false", "0", "disable"):
            _toggle_config(gateway, "llm_classifier_enabled", False)
            return _reply_or_rewrite(gateway, source, "📋 Smart Router classifier: **regex only**")
        elif arg:
            return _reply_or_rewrite(gateway, source, _format_classification_preview(arg, cfg, title="📋 Smart Router classifier"))
        else:
            return _reply_or_rewrite(gateway, source, f"Usage: `/smart-router classifier llm|regex` or `/smart-router classifier <message>`\nCurrent: {'llm' if cfg.llm_classifier_enabled else 'regex'}")

    elif subcommand == "help":
        # Already handled by commands.py above — keep fallback
        return _SKIP

    return _SKIP


def _toggle_config(gateway: Any, key: str, value: Any) -> None:
    """Toggle a runtime config override on the gateway."""
    if not hasattr(gateway, "_smart_router_runtime_config"):
        setattr(gateway, "_smart_router_runtime_config", {})
    getattr(gateway, "_smart_router_runtime_config")[key] = value


def _reply_or_rewrite(gateway: Any, source: Any, message: str) -> dict:
    """Try to send a direct reply via adapter, fall back to rewrite.

    Returns SKIP if direct send works, otherwise returns rewrite action
    so the agent processes the message as a normal prompt.
    """
    adapter = gateway.adapters.get(source.platform) if hasattr(gateway, "adapters") else None
    if adapter and hasattr(adapter, "send"):
        try:
            import asyncio
            chat_id = getattr(source, "chat_id", None)
            if chat_id:
                asyncio.ensure_future(adapter.send(chat_id, message))
                return _SKIP
        except Exception:
            logger.debug("Could not send direct reply, falling back to rewrite")
    return {"action": "rewrite", "text": message}


def _format_classification_preview(text: str, cfg: Any, title: str = "Smart Router") -> str:
    """Return a human-readable route preview without applying an override."""
    if cfg.llm_classifier_enabled:
        classification = llm_classify_message(
            text,
            provider=cfg.llm_classifier_provider,
            model=cfg.llm_classifier_model,
            scoring=cfg.scoring,
            patterns=cfg.patterns,
        )
        mode = "llm"
    else:
        classification = classify_message(text, scoring=cfg.scoring, patterns=cfg.patterns)
        mode = "regex"

    route = cfg.route_for_score(classification.score)
    return (
        f"{title}\n"
        f"{route.emoji} {route.name} → `{route.provider}/{route.model}` (score: {classification.score})\n"
        f"Mode: {mode}\n"
        f"Reason: {classification.reason}"
    )


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------

_KNOWN_ENV_KEYS = {
    "opencode-go": "OPENCODE_GO_API_KEY",
    "openai-codex": "OPENAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "nous": "NOUS_API_KEY",
}

_KNOWN_BASE_URLS = {
    "opencode-go": "https://opencode.ai/zen/go/v1",
    "openai-codex": "https://chatgpt.com/backend-api/codex",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "nous": "https://inference-api.nousresearch.com/v1",
}

_KNOWN_API_MODES = {
    "openai-codex": "codex_responses",
}


def _resolve_route_runtime(route: Any) -> dict:
    """Resolve provider runtime fields for the *target* provider.

    The Hermes gateway resolves runtime credentials once from the GLOBAL
    provider (e.g. openai-codex) via ``_resolve_runtime_agent_kwargs()``.
    When a session override changes the provider but the override has no
    ``api_key``, the gateway keeps the global provider's stale credentials
    — the agent ends up with ``provider=nous`` but
    ``api_key=<openai-codex token>`` and ``base_url=<openai-codex URL>``.

    This function calls ``resolve_runtime_provider(requested=route.provider)``
    so that api_key, base_url, and api_mode all match the ROUTED provider.
    All three fields are always included in the returned dict — even if
    api_key is empty — so that ``_apply_session_model_override`` overwrites
    the global provider's stale values.
    """
    provider = route.provider
    resolved_api_key = None
    resolved_base_url = None
    resolved_api_mode = None

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        _r = resolve_runtime_provider(
            requested=provider,
            explicit_base_url=route.base_url,
            explicit_api_key=route.api_key,
        )
        resolved_api_key = _r.get("api_key") or None
        resolved_base_url = _r.get("base_url") or None
        resolved_api_mode = _r.get("api_mode") or None
    except Exception:
        logger.debug(
            "resolve_runtime_provider failed for provider=%s; using env / static fallbacks",
            provider, exc_info=True,
        )

    # api_key: explicit route config > runtime_provider > env var
    api_key = route.api_key or resolved_api_key
    if not api_key:
        env_key = _KNOWN_ENV_KEYS.get(provider)
        if env_key:
            api_key = os.getenv(env_key)

    # base_url MUST point at the TARGET provider.  A stale base_url from
    # the global provider (e.g. openai-codex's URL) combined with
    # provider="nous" causes API calls to fail with 404 / auth errors.
    base_url = (
        route.base_url
        or resolved_base_url
        or _KNOWN_BASE_URLS.get(provider)
        or ""
    )
    api_mode = (
        route.api_mode
        or resolved_api_mode
        or _KNOWN_API_MODES.get(provider)
        or "chat_completions"
    )

    # Always include all three keys — even empty — so the gateway
    # overwrites the global provider's stale values instead of merging.
    return {
        "api_key": api_key or "",
        "base_url": base_url,
        "api_mode": api_mode,
    }


# ---------------------------------------------------------------------------
# Config with runtime overrides
# ---------------------------------------------------------------------------

def _get_effective_config(gateway: Any) -> Any:
    """Load config and apply runtime overrides from slash commands."""
    cfg = load_config(gateway)
    runtime = getattr(gateway, "_smart_router_runtime_config", {})
    if not runtime:
        return cfg

    # Apply runtime overrides by creating a modified config
    from dataclasses import replace
    for key, value in runtime.items():
        if hasattr(cfg, key):
            cfg = replace(cfg, **{key: value})
    return cfg


# ---------------------------------------------------------------------------
# Main hook: route_gateway_message
# ---------------------------------------------------------------------------

def route_gateway_message(event: Any = None, gateway: Any = None,
                          session_store: Any = None, **kwargs: Any) -> dict:
    """Route one inbound gateway message to a provider/model.

    This hook must fail open. If anything about the Hermes version or gateway
    payload differs from what we expect, we return allow and let Hermes use its
    normal model/fallback configuration.
    """
    try:
        if event is None or gateway is None:
            return _ALLOW
        if getattr(event, "internal", False):
            return _ALLOW

        text = getattr(event, "text", None) or ""
        source = getattr(event, "source", None)

        # --- Slash command handling ---
        command_result = _handle_slash_command(event, gateway)
        if command_result is not None:
            return command_result

        if not text.strip() or source is None:
            return _ALLOW

        cfg = _get_effective_config(gateway)
        if not cfg.enabled:
            return _ALLOW

        overrides = getattr(gateway, "_session_model_overrides", None)
        if not isinstance(overrides, dict):
            logger.debug("Smart router disabled: gateway has no _session_model_overrides dict")
            return _ALLOW

        session_key = _safe_get_session_key(gateway, source)
        if not session_key:
            logger.debug("Smart router disabled: could not derive session key")
            return _ALLOW

        existing = overrides.get(session_key)
        if cfg.respect_manual_override and _is_manual_override(existing):
            logger.info("Smart router respecting existing manual /model override for session %s", session_key)
            return _ALLOW

        # Classification — optionally use LLM for uncertain cases
        if cfg.llm_classifier_enabled:
            classification = llm_classify_message(
                text,
                provider=cfg.llm_classifier_provider,
                model=cfg.llm_classifier_model,
                scoring=cfg.scoring,
                patterns=cfg.patterns,
            )
        else:
            classification = classify_message(text, scoring=cfg.scoring, patterns=cfg.patterns)

        # Route by score — uses the new parametrizable score-range model
        route = cfg.route_for_score(classification.score)
        override = route.as_override()
        runtime = _resolve_route_runtime(route)
        # Always include api_key/base_url/api_mode in the override,
        # even if empty.  The Hermes gateway skips None values in
        # _apply_session_model_override but NOT empty strings — an
        # empty api_key forces the agent to resolve credentials fresh
        # for the target provider instead of reusing the global one.
        for key in ("api_key", "base_url", "api_mode"):
            override[key] = runtime.get(key, "")
        override["_smart_router_reason"] = classification.reason
        override["_smart_router_score"] = classification.score

        # Store routing decision for the footer hook
        decisions = _get_decisions_store(gateway)
        decisions[session_key] = {
            "complexity": route.name,
            "provider": route.provider,
            "model": route.model,
            "score": classification.score,
            "reason": classification.reason,
            "emoji": route.emoji,
        }

        if cfg.dry_run:
            logger.info(
                "Smart router dry-run: session=%s complexity=%s route=%s provider=%s model=%s score=%s",
                session_key,
                classification.complexity,
                route.name,
                route.provider,
                route.model,
                classification.score,
            )
            return _ALLOW

        overrides[session_key] = override

        evict = getattr(gateway, "_evict_cached_agent", None)
        if callable(evict):
            evict(session_key)

        logger.info(
            "Smart router applied: session=%s complexity=%s route=%s provider=%s model=%s score=%s",
            session_key,
            classification.complexity,
            route.name,
            route.provider,
            route.model,
            classification.score,
        )
        return _ALLOW
    except Exception:
        logger.exception("Smart router failed; allowing Hermes default routing")
        return _ALLOW


# ---------------------------------------------------------------------------
# Footer hook: transform_llm_output
# ---------------------------------------------------------------------------

def transform_llm_output(response_text: str = "", session_id: str = "",
                          model: str = "", platform: str = "",
                          **kwargs: Any) -> Optional[str]:
    """Append a routing footer to the LLM response.

    This hook reads the routing decision stored by route_gateway_message
    and appends a small italicized footer showing which route was chosen.
    """
    try:
        if not response_text:
            return None

        gateway = kwargs.get("gateway")
        if gateway is None:
            return None

        cfg = _get_effective_config(gateway)
        if not cfg.show_route_footer:
            return None

        decisions = _get_decisions_store(gateway)
        if session_id not in decisions:
            return None

        info = decisions.pop(session_id)  # One-shot: clean up after reading
        route_name = info["complexity"]
        provider = info["provider"]
        route_model = info["model"]
        score = info["score"]
        emoji = info.get("emoji", "⚪")

        footer = f"\n\n---\n{emoji} *Smart Router: {route_name} → `{provider}/{route_model}` (score: {score})*"

        return response_text + footer
    except Exception:
        logger.debug("Smart router footer hook failed (non-fatal)")
        return None
