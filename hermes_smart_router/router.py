"""Gateway hook implementation for Hermes Smart Router."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .classifier import classify_message
from .config import load_config

logger = logging.getLogger(__name__)

_ALLOW = {"action": "allow"}


def _safe_get_session_key(gateway: Any, source: Any) -> Optional[str]:
    session_key_fn = getattr(gateway, "_session_key_for_source", None)
    if callable(session_key_fn):
        key = session_key_fn(source)
        if isinstance(key, str) and key:
            return key
    return None


def _is_manual_override(existing: Any) -> bool:
    return isinstance(existing, dict) and bool(existing) and not existing.get("_smart_router")


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
        if not text.strip() or source is None:
            return _ALLOW

        cfg = load_config(gateway)
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

        classification = classify_message(text)
        route = cfg.route_for(classification.complexity)
        override = route.as_override()
        override["_smart_router_reason"] = classification.reason
        override["_smart_router_score"] = classification.score

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
