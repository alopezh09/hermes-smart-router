"""Gateway hook implementation for Hermes Smart Router."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .classifier import classify_message, llm_classify_message
from .config import load_config

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


def _handle_slash_command(event: Any, gateway: Any) -> Optional[dict]:
    """Handle /smart-router slash commands. Returns action dict or None."""
    text = (event.text or "").strip()
    if not text.startswith("/smart-router"):
        return None

    source = getattr(event, "source", None)
    if source is None:
        return _SKIP

    cfg = load_config(gateway)
    parts = text.split(maxsplit=1)
    subcommand = parts[1].strip() if len(parts) > 1 else ""

    if subcommand == "status" or subcommand == "":
        lines = [
            "**Smart Router Status**",
            f"• Enabled: {'✅' if cfg.enabled else '❌'}",
            f"• Dry-run: {'🔬' if cfg.dry_run else '🚀'}",
            f"• Show footer: {'✅' if cfg.show_route_footer else '❌'}",
            f"• LLM classifier: {'🤖' if cfg.llm_classifier_enabled else '📋 (regex)'}",
            f"• Respect manual /model: {'✅' if cfg.respect_manual_override else '❌'}",
            "",
            "**Routes:**",
        ]
        for complexity in ("simple", "medium", "complex"):
            route = cfg.routes.get(complexity)
            if route:
                icon = {"simple": "🟢", "medium": "🟡", "complex": "🔴"}.get(complexity, "⚪")
                lines.append(f"  {icon} {complexity}: `{route.provider}/{route.model}`")

        # Try to send response directly via adapter
        adapter = gateway.adapters.get(source.platform) if hasattr(gateway, "adapters") else None
        if adapter and hasattr(adapter, "send"):
            try:
                import asyncio
                chat_id = getattr(source, "chat_id", None)
                if chat_id:
                    asyncio.ensure_future(
                        adapter.send(chat_id, "\n".join(lines))
                    )
                    return _SKIP
            except Exception:
                logger.debug("Could not send status via adapter, falling back to rewrite")

        # Fallback: rewrite to a prompt the agent can answer
        return {"action": "rewrite", "text": f"Show me the current Smart Router status:\n\n" + "\n".join(lines)}

    elif subcommand.startswith("dry-run"):
        arg = subcommand[len("dry-run"):].strip().lower()
        if arg in ("on", "true", "1", "enable"):
            _toggle_config(gateway, "dry_run", True)
            return _reply_or_rewrite(gateway, source, "🔬 Smart Router dry-run mode **ON** — routing decisions will be logged but not applied.")
        elif arg in ("off", "false", "0", "disable"):
            _toggle_config(gateway, "dry_run", False)
            return _reply_or_rewrite(gateway, source, "🚀 Smart Router dry-run mode **OFF** — routing will be applied normally.")
        else:
            return _reply_or_rewrite(gateway, source, f"Usage: `/smart-router dry-run on|off`\nCurrent: {'on' if cfg.dry_run else 'off'}")

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
        else:
            return _reply_or_rewrite(gateway, source, f"Usage: `/smart-router classifier llm|regex`\nCurrent: {'llm' if cfg.llm_classifier_enabled else 'regex'}")

    elif subcommand == "help":
        help_text = (
            "**Smart Router Commands**\n"
            "• `/smart-router` — show status\n"
            "• `/smart-router dry-run on|off` — toggle dry-run\n"
            "• `/smart-router footer on|off` — toggle route footer\n"
            "• `/smart-router classifier llm|regex` — switch classifier mode\n"
            "• `/smart-router help` — this help"
        )
        return _reply_or_rewrite(gateway, source, help_text)

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


def route_gateway_message(event: Any = None, gateway: Any = None, session_store: Any = None, **kwargs: Any) -> dict:
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
            )
        else:
            classification = classify_message(text)

        route = cfg.route_for(classification.complexity)
        override = route.as_override()
        override["_smart_router_reason"] = classification.reason
        override["_smart_router_score"] = classification.score

        # Store routing decision for the footer hook
        decisions = _get_decisions_store(gateway)
        decisions[session_key] = {
            "complexity": classification.complexity,
            "provider": route.provider,
            "model": route.model,
            "score": classification.score,
            "reason": classification.reason,
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
            # Try to get the gateway from the global plugin context
            return None

        cfg = _get_effective_config(gateway)
        if not cfg.show_route_footer:
            return None

        decisions = _get_decisions_store(gateway)
        if session_id not in decisions:
            return None

        info = decisions.pop(session_id)  # One-shot: clean up after reading
        complexity = info["complexity"]
        provider = info["provider"]
        route_model = info["model"]
        score = info["score"]

        emoji = {"simple": "🟢", "medium": "🟡", "complex": "🔴"}.get(complexity, "⚪")
        footer = f"\n\n---\n{emoji} *Smart Router: {complexity} → `{provider}/{route_model}` (score: {score})*"

        return response_text + footer
    except Exception:
        logger.debug("Smart router footer hook failed (non-fatal)")
        return None
