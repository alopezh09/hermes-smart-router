"""Gateway hook implementation for Hermes Smart Router."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, List

from .classifier import classify_message, llm_classify_message
from .config import load_config, Route, ScoringConfig, PatternConfig
from .providers import resolve_api_key, resolve_base_url, resolve_api_mode

logger = logging.getLogger(__name__)

_ALLOW = {"action": "allow"}
_SKIP = {"action": "skip"}

# Store routing decisions on the gateway object so the transform_llm_output
# hook can read them. Key: session_key, Value: {"ts": ..., "info": ...}.
_ROUTING_DECISIONS_ATTR = "_smart_router_decisions"

# ── Improvement #4: max age for stale decisions (seconds) ──
_DECISION_MAX_AGE_S = 300  # 5 minutes

# ── Bug #3 + Mejora #3: persistent runtime config path ──
_RUNTIME_STATE_PATH = Path(os.path.expanduser("~/.hermes/smart_router_state.json"))

__version__ = "0.2.0"


# ============================================================================
# Improvement #2 — Usage metrics
# ============================================================================

_METRICS_ATTR = "_smart_router_metrics"


def _get_metrics_store(gateway: Any) -> dict:
    if not hasattr(gateway, _METRICS_ATTR):
        setattr(gateway, _METRICS_ATTR, {"routes": {}, "total": 0, "started_at": time.time()})
    return getattr(gateway, _METRICS_ATTR)


def _record_route_hit(gateway: Any, route_name: str) -> None:
    """Increment the hit counter for *route_name*."""
    metrics = _get_metrics_store(gateway)
    metrics["total"] += 1
    metrics["routes"][route_name] = metrics["routes"].get(route_name, 0) + 1


def get_metrics_summary(gateway: Any) -> dict:
    """Return a copy of current metrics for display or testing."""
    import copy
    return copy.deepcopy(_get_metrics_store(gateway))


# ============================================================================
# Bug #3 + Mejora #3 — Persistent runtime config
# ============================================================================

_RUNTIME_CONFIG_ATTR = "_smart_router_runtime_config"


def _load_runtime_state() -> dict:
    """Load persisted runtime toggles from disk."""
    try:
        if _RUNTIME_STATE_PATH.exists():
            return json.loads(_RUNTIME_STATE_PATH.read_text())
    except Exception:
        logger.debug("Could not load smart router runtime state", exc_info=True)
    return {}


def _save_runtime_state(state: dict) -> None:
    """Persist runtime toggles to disk."""
    try:
        _RUNTIME_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RUNTIME_STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception:
        logger.debug("Could not save smart router runtime state", exc_info=True)


def _get_runtime_config(gateway: Any) -> dict:
    """Get runtime toggles, loading from disk on first access."""
    if not hasattr(gateway, _RUNTIME_CONFIG_ATTR):
        persisted = _load_runtime_state()
        setattr(gateway, _RUNTIME_CONFIG_ATTR, persisted)
    return getattr(gateway, _RUNTIME_CONFIG_ATTR)


def _toggle_config(gateway: Any, key: str, value: Any) -> None:
    """Toggle a runtime config override and persist it."""
    cfg = _get_runtime_config(gateway)
    cfg[key] = value
    _save_runtime_state(cfg)


# ============================================================================
# Improvement #4 — Clean stale decisions
# ============================================================================

def _get_decisions_store(gateway: Any) -> dict:
    """Get or create the routing decisions dict, cleaning stale entries."""
    if not hasattr(gateway, _ROUTING_DECISIONS_ATTR):
        setattr(gateway, _ROUTING_DECISIONS_ATTR, {})
    store = getattr(gateway, _ROUTING_DECISIONS_ATTR)

    # Purge entries older than _DECISION_MAX_AGE_S
    now = time.time()
    stale = [k for k, v in store.items()
             if isinstance(v, dict) and now - v.get("ts", 0) > _DECISION_MAX_AGE_S]
    for k in stale:
        del store[k]

    return store


def _store_decision(gateway: Any, session_key: str, info: dict) -> None:
    """Store a routing decision with a timestamp for the footer hook."""
    store = _get_decisions_store(gateway)
    store[session_key] = {"ts": time.time(), "info": info}


# ============================================================================
# Helpers
# ============================================================================

def _safe_get_session_key(gateway: Any, source: Any) -> Optional[str]:
    session_key_fn = getattr(gateway, "_session_key_for_source", None)
    if callable(session_key_fn):
        key = session_key_fn(source)
        if isinstance(key, str) and key:
            return key
    return None


def _is_manual_override(existing: Any) -> bool:
    return isinstance(existing, dict) and bool(existing) and not existing.get("_smart_router")


# ============================================================================
# Slash command handling
# ============================================================================

def _handle_slash_command(event: Any, gateway: Any) -> Optional[dict]:
    """Handle /smart-router slash commands. Returns action dict or None."""
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
        if subcommand.startswith("dry-run") and subcommand[len("dry-run"):].strip().lower() not in (
            "on", "true", "1", "enable", "off", "false", "0", "disable", "",
        ):
            pass  # dry-run with a message: just display
        elif subcommand.startswith("classifier") and subcommand[len("classifier"):].strip().lower() not in (
            "llm", "on", "true", "1", "enable", "regex", "off", "false", "0", "disable", "",
        ):
            pass  # classifier with a message: just display
        elif subcommand.startswith("test") and subcommand[len("test"):].strip():
            pass  # test with a message: just display
        return _reply_or_rewrite(gateway, source, response)

    # ── Toggle commands (modify runtime state) ──
    if subcommand == "status" or subcommand == "":
        return _SKIP

    elif subcommand.startswith("dry-run"):
        arg = subcommand[len("dry-run"):].strip().lower()
        if arg in ("on", "true", "1", "enable"):
            _toggle_config(gateway, "dry_run", True)
            return _reply_or_rewrite(gateway, source,
                "🔬 Smart Router dry-run mode **ON** — routing decisions will be logged but not applied.")
        elif arg in ("off", "false", "0", "disable"):
            _toggle_config(gateway, "dry_run", False)
            return _reply_or_rewrite(gateway, source,
                "🚀 Smart Router dry-run mode **OFF** — routing will be applied normally.")
        elif arg:
            return _reply_or_rewrite(gateway, source,
                _format_classification_preview(arg, cfg, title="🔬 Smart Router dry-run"))
        else:
            return _reply_or_rewrite(gateway, source,
                f"Usage: `/smart-router dry-run on|off` or `/smart-router dry-run <message>`\n"
                f"Current: {'on' if cfg.dry_run else 'off'}")

    elif subcommand.startswith("footer"):
        arg = subcommand[len("footer"):].strip().lower()
        if arg in ("on", "true", "1", "enable"):
            _toggle_config(gateway, "show_route_footer", True)
            return _reply_or_rewrite(gateway, source,
                "✅ Smart Router footer **ON** — route info will appear at the end of each response.")
        elif arg in ("off", "false", "0", "disable"):
            _toggle_config(gateway, "show_route_footer", False)
            return _reply_or_rewrite(gateway, source,
                "❌ Smart Router footer **OFF** — route info will be hidden.")
        else:
            return _reply_or_rewrite(gateway, source,
                f"Usage: `/smart-router footer on|off`\n"
                f"Current: {'on' if cfg.show_route_footer else 'off'}")

    elif subcommand.startswith("classifier"):
        arg = subcommand[len("classifier"):].strip().lower()
        if arg in ("llm", "on", "true", "1", "enable"):
            _toggle_config(gateway, "llm_classifier_enabled", True)
            return _reply_or_rewrite(gateway, source,
                "🤖 Smart Router classifier: **LLM** (with regex fallback)")
        elif arg in ("regex", "off", "false", "0", "disable"):
            _toggle_config(gateway, "llm_classifier_enabled", False)
            return _reply_or_rewrite(gateway, source,
                "📋 Smart Router classifier: **regex only**")
        elif arg:
            return _reply_or_rewrite(gateway, source,
                _format_classification_preview(arg, cfg, title="📋 Smart Router classifier"))
        else:
            return _reply_or_rewrite(gateway, source,
                f"Usage: `/smart-router classifier llm|regex` or `/smart-router classifier <message>`\n"
                f"Current: {'llm' if cfg.llm_classifier_enabled else 'regex'}")

    # ── Mejora #5: /smart-router test <msg> ──
    elif subcommand.startswith("test"):
        arg = subcommand[len("test"):].strip()
        if arg:
            return _reply_or_rewrite(gateway, source,
                _format_detailed_test(arg, cfg))
        else:
            return _reply_or_rewrite(gateway, source,
                "Usage: `/smart-router test <message>`\n\n"
                "Shows the full classification breakdown: score, weights, tier resolved, "
                "and the list of all routes that would match or not match.")

    # ── Mejora #2: metrics ──
    elif subcommand == "metrics":
        return _reply_or_rewrite(gateway, source, _format_metrics(gateway, cfg))

    # ── Mejora #9: version ──
    elif subcommand == "version":
        return _reply_or_rewrite(gateway, source,
            f"**Smart Router** v{__version__}\n"
            f"Routes: {cfg.route_names}\n"
            f"Classifier: {'🤖 LLM' if cfg.llm_classifier_enabled else '📋 regex'}")

    elif subcommand == "help":
        return _SKIP

    return _SKIP


# ============================================================================
# Improvement #5 — Detailed test breakdown
# ============================================================================

def _format_detailed_test(text: str, cfg: Any) -> str:
    """Return a detailed classification breakdown for /smart-router test <msg>."""
    from .classifier import _count_matches

    lines = ["🧪 **Smart Router — Test Classification**", ""]

    # 1. Classify
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

    # 2. Score breakdown
    s = cfg.scoring; p = cfg.patterns
    normalized = " ".join(text.strip().split())
    words = normalized.split()
    word_count = len(words)
    has_code_block = "```" in normalized
    has_many_requirements = sum(
        token in normalized
        for token in ["\n-", "\n*", " 1.", " 2.", " 3.",
                      "1)", "2)", "3)", ";", " y ", " and ",
                      " también ", " además ", " also ", " plus "]
    )
    simple_hits = _count_matches(p.simple, normalized)
    medium_hits = _count_matches(p.medium, normalized)
    complex_hits = _count_matches(p.complex, normalized)

    lines.append("**📊 Score breakdown:**")
    lines.append(f"• Complex patterns matched: **{complex_hits}** × {s.weight_complex_pattern} = {complex_hits * s.weight_complex_pattern}")
    lines.append(f"• Medium patterns matched: **{medium_hits}** × {s.weight_medium_pattern} = {medium_hits * s.weight_medium_pattern}")
    lines.append(f"• Simple patterns matched: **{simple_hits}** × {s.weight_simple_pattern} = {simple_hits * s.weight_simple_pattern}")
    lines.append(f"• Code blocks: {'✅' if has_code_block else '❌'} {'(' + str(s.weight_code_block) + ')' if has_code_block else ''}")

    if word_count >= 80:
        lines.append(f"• Very long ({word_count} words): +{s.weight_very_long}")
    elif word_count >= 35:
        lines.append(f"• Long ({word_count} words): +{s.weight_long}")
    lines.append(f"• Requirement list markers: **{min(has_many_requirements, 3)}** × {s.weight_requirement_list}")
    lines.append(f"• **Total score: {classification.score}**")
    lines.append("")

    # 3. Route matching
    route = cfg.route_for_score(classification.score)
    lines.append(f"**🎯 Resolved route:** {route.emoji} `{route.name}` → `{route.provider}/{route.model}`")
    lines.append(f"• Score range: {route.min_score}–{route.max_score}")
    lines.append(f"• Reason: {classification.reason}")
    lines.append(f"• Classifier: {mode}")
    lines.append("")

    # 4. All routes & their score match
    lines.append("**🗺️ Route map:**")
    for r in cfg.routes:
        match = "✅" if r.matches_score(classification.score) else "❌"
        lines.append(f"  {match} {r.emoji} `{r.name}` ({r.min_score}–{r.max_score}) → `{r.provider}/{r.model}`")

    # 5. Escalation info (Mejora #7)
    next_route = _find_next_tier(cfg, classification.score)
    if next_route:
        lines.append("")
        lines.append(f"**⏫ Escalation:** If `{route.name}` fails → `{next_route.name}`")

    return "\n".join(lines)


# ============================================================================
# Mejora #2 — Metrics formatting
# ============================================================================

def _format_metrics(gateway: Any, cfg: Any) -> str:
    """Format usage metrics for display."""
    m = get_metrics_summary(gateway)
    uptime = time.time() - m.get("started_at", time.time())
    hours, rem = divmod(int(uptime), 3600)
    mins, secs = divmod(rem, 60)

    lines = ["📊 **Smart Router — Metrics**", ""]
    lines.append(f"• Total routed messages: **{m['total']}**")
    lines.append(f"• Uptime: {hours}h {mins}m {secs}s")
    lines.append("")
    lines.append("**Per-route hits:**")

    routes = m.get("routes", {})
    if routes:
        for name, count in sorted(routes.items(), key=lambda x: -x[1]):
            route = None
            for r in cfg.routes:
                if r.name == name:
                    route = r
                    break
            emoji = route.emoji if route else "⚪"
            pct = (count / m["total"] * 100) if m["total"] else 0
            bar_len = min(int(pct / 5), 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {emoji} `{name}`: {count} ({pct:.0f}%) {bar}")
    else:
        lines.append("  _(no messages routed yet)_")

    return "\n".join(lines)


# ============================================================================
# Mejora #7 — Escalation helper
# ============================================================================

def _find_next_tier(cfg: Any, current_score: int) -> Optional[Route]:
    """Find the next higher-scoring route for escalation on failure."""
    current_route = cfg.route_for_score(current_score)
    for r in cfg.routes:
        if r.min_score > current_route.max_score:
            return r
    return None


def _apply_escalation_route(
    gateway: Any,
    session_key: str,
    original_route: Route,
    classification,
    cfg: Any,
) -> bool:
    """Attempt to escalate to the next tier.

    Returns True if an escalation route was applied, False otherwise.
    This is called when a user-visible error suggests the current tier's
    model failed (detected via the ``_smart_router_error`` marker on the
    session override).
    """
    next_route = _find_next_tier(cfg, classification.score)
    if next_route is None:
        return False

    logger.info(
        "Smart router escalating: session=%s from=%s to=%s",
        session_key, original_route.name, next_route.name,
    )

    override = next_route.as_override()
    runtime = _resolve_route_runtime(next_route)
    for key in ("api_key", "base_url", "api_mode"):
        override[key] = runtime.get(key, "")
    override["_smart_router_reason"] = classification.reason
    override["_smart_router_score"] = classification.score
    override["_smart_router_escalated_from"] = original_route.name

    overrides = getattr(gateway, "_session_model_overrides", None)
    if isinstance(overrides, dict):
        overrides[session_key] = override

    evict = getattr(gateway, "_evict_cached_agent", None)
    if callable(evict):
        evict(session_key)

    _record_route_hit(gateway, next_route.name)
    return True


def check_and_escalate_if_needed(
    gateway: Any,
    session_key: str,
    last_error: Optional[str] = None,
) -> bool:
    """Public API: check session override for escalation markers and escalate.

    This is meant to be called from outside (e.g. after an agent error)
    to check if the current session should be escalated to a higher tier.

    Returns True if escalation occurred.
    """
    try:
        overrides = getattr(gateway, "_session_model_overrides", None)
        if not isinstance(overrides, dict):
            return False

        override = overrides.get(session_key)
        if not isinstance(override, dict):
            return False

        # Only escalate if this is a smart-router override that had an error
        if not override.get("_smart_router"):
            return False
        if not override.get("_smart_router_error") and not last_error:
            return False

        cfg = _get_effective_config(gateway)
        score = override.get("_smart_router_score", 0)

        # Find original route
        route_name = override.get("_smart_router_route", "")
        original_route = None
        for r in cfg.routes:
            if r.name == route_name:
                original_route = r
                break
        if original_route is None:
            return False

        # Create a simple classification-like object
        from dataclasses import dataclass
        @dataclass
        class _SimpleClass:
            score: int
            complexity: str
            reason: str

        classification = _SimpleClass(
            score=score,
            complexity=route_name,
            reason=override.get("_smart_router_reason", "escalation"),
        )

        return _apply_escalation_route(gateway, session_key, original_route, classification, cfg)
    except Exception:
        logger.debug("Smart router escalation check failed (non-fatal)", exc_info=True)
        return False


# ============================================================================
# Reply helper
# ============================================================================

def _reply_or_rewrite(gateway: Any, source: Any, message: str) -> dict:
    """Try to send a direct reply via adapter, fall back to rewrite."""
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


# ============================================================================
# Runtime resolution
# ============================================================================

def _resolve_route_runtime(route: Any) -> dict:
    """Resolve provider runtime fields for the *target* provider."""
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

    api_key = route.api_key or resolved_api_key or resolve_api_key(provider) or ""
    base_url = route.base_url or resolved_base_url or resolve_base_url(provider)
    api_mode = route.api_mode or resolved_api_mode or resolve_api_mode(provider)

    return {"api_key": api_key, "base_url": base_url, "api_mode": api_mode}


# ============================================================================
# Config with runtime overrides
# ============================================================================

def _get_effective_config(gateway: Any) -> Any:
    """Load config and apply runtime overrides from slash commands + persisted state."""
    cfg = load_config(gateway)
    runtime = _get_runtime_config(gateway)
    if not runtime:
        return cfg

    from dataclasses import replace
    for key, value in runtime.items():
        if hasattr(cfg, key):
            cfg = replace(cfg, **{key: value})
    return cfg


# ============================================================================
# Main hook: route_gateway_message
# ============================================================================

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

        # ── Mejora #7: check if last call had an error marker ──
        if isinstance(existing, dict) and existing.get("_smart_router") and existing.get("_smart_router_error"):
            logger.info("Smart router detected error marker on session %s; escalating", session_key)
            if check_and_escalate_if_needed(gateway, session_key):
                return _ALLOW

        # Classification
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

        route = cfg.route_for_score(classification.score)
        override = route.as_override()
        runtime = _resolve_route_runtime(route)
        for key in ("api_key", "base_url", "api_mode"):
            override[key] = runtime.get(key, "")
        override["_smart_router_reason"] = classification.reason
        override["_smart_router_score"] = classification.score

        # ── Store decision for footer (with timestamp) ──
        _store_decision(gateway, session_key, {
            "complexity": route.name,
            "provider": route.provider,
            "model": route.model,
            "score": classification.score,
            "reason": classification.reason,
            "emoji": route.emoji,
        })

        # ── Mejora #2: record metrics ──
        if not cfg.dry_run:
            _record_route_hit(gateway, route.name)

        if cfg.dry_run:
            logger.info(
                "Smart router dry-run: session=%s route=%s provider=%s model=%s score=%s",
                session_key, route.name, route.provider, route.model, classification.score,
            )
            return _ALLOW

        overrides[session_key] = override

        evict = getattr(gateway, "_evict_cached_agent", None)
        if callable(evict):
            evict(session_key)

        logger.info(
            "Smart router applied: session=%s route=%s provider=%s model=%s score=%s",
            session_key, route.name, route.provider, route.model, classification.score,
        )
        return _ALLOW
    except Exception:
        logger.exception("Smart router failed; allowing Hermes default routing")
        return _ALLOW


# ============================================================================
# Footer hook: transform_llm_output
# ============================================================================

def transform_llm_output(response_text: str = "", session_id: str = "",
                          model: str = "", platform: str = "",
                          **kwargs: Any) -> Optional[str]:
    """Append a routing footer to the LLM response."""
    try:
        if not response_text:
            return None

        gateway = kwargs.get("gateway")
        if gateway is None:
            return None

        cfg = _get_effective_config(gateway)
        if not cfg.show_route_footer:
            return None

        store = _get_decisions_store(gateway)
        entry = store.pop(session_id, None)
        if entry is None:
            return None

        info = entry["info"]
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
