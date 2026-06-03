"""Tests for new v0.2.0 features: metrics, test command, version, escalation."""

import time
from types import SimpleNamespace

from hermes_smart_router.router import (
    route_gateway_message,
    _record_route_hit,
    get_metrics_summary,
    _get_runtime_config,
    _toggle_config,
    _format_detailed_test,
    _format_metrics,
    check_and_escalate_if_needed,
    __version__,
)


class FakeAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, message):
        self.sent.append((chat_id, message))


class FakeGateway:
    def __init__(self, config=None, adapters=None):
        self.config = config or {}
        self._session_model_overrides = {}
        self.evicted = []
        self.adapters = adapters or {}

    def _session_key_for_source(self, source):
        return f"{source.platform}:{source.chat_id}:{source.user_id}"

    def _evict_cached_agent(self, session_key):
        self.evicted.append(session_key)


def event(text="hola"):
    source = SimpleNamespace(platform="telegram", chat_id="1", user_id="2")
    return SimpleNamespace(text=text, source=source, internal=False)


# =============================================================================
# Metrics (Mejora #2)
# =============================================================================

def test_metrics_count_route_hits():
    gateway = FakeGateway()
    _record_route_hit(gateway, "simple")
    _record_route_hit(gateway, "simple")
    _record_route_hit(gateway, "complex")

    m = get_metrics_summary(gateway)
    assert m["total"] == 3
    assert m["routes"]["simple"] == 2
    assert m["routes"]["complex"] == 1
    assert "started_at" in m


def test_metrics_slash_command():
    """Metrics command returns a formatted message."""
    gateway = FakeGateway()
    # Route some messages first
    route_gateway_message(event("hola"), gateway)
    route_gateway_message(event("gracias"), gateway)

    # Check metrics via slash command
    result = route_gateway_message(event("/smart-router metrics"), gateway)
    assert "action" in result
    text = result.get("text", "")
    assert "Total" in text or "total" in text.lower()


def test_metrics_includes_uptime():
    """Metrics output contains uptime info."""
    gateway = FakeGateway()
    _record_route_hit(gateway, "simple")
    output = _format_metrics(gateway, load_config_for_test())
    assert "Uptime" in output
    assert "simple" in output


# =============================================================================
# Test command (Mejora #5)
# =============================================================================

def test_test_slash_command_returns_detailed_breakdown():
    gateway = FakeGateway()
    result = route_gateway_message(
        event("/smart-router test implementa un compilador"), gateway)
    assert result["action"] == "rewrite"
    text = result["text"]
    assert "Test Classification" in text
    assert "Score breakdown" in text
    assert "Total score" in text


def test_detailed_test_includes_escalation_info():
    """Detailed test output mentions next escalation tier for medium messages."""
    cfg = load_config_for_test()
    output = _format_detailed_test("explícame cómo funciona OAuth2 con ejemplos", cfg)
    assert "Escalation" in output or "⏫" in output


# =============================================================================
# Version command (Mejora #9)
# =============================================================================

def test_version_slash_command():
    gateway = FakeGateway()
    result = route_gateway_message(event("/smart-router version"), gateway)
    text = result.get("text", "")
    assert "Smart Router" in text
    assert __version__ in text
    assert "routes" in text.lower()


# =============================================================================
# Escalation (Mejora #7)
# =============================================================================

def test_escalation_finds_next_tier():
    """When medium route fails, it should escalate to complex."""
    gateway = FakeGateway()
    key = "telegram:1:2"

    # Route a medium message
    route_gateway_message(event("explícame cómo funciona OAuth2"), gateway)
    override = gateway._session_model_overrides[key]
    assert override["provider"] == "opencode-go"

    # Mark as errored
    override["_smart_router_error"] = True
    gateway._session_model_overrides[key] = override

    # Next message should trigger escalation
    route_gateway_message(event("explícame SSL/TLS también"), gateway)
    escalated = gateway._session_model_overrides[key]
    # Should have escalated to complex tier
    assert escalated["provider"] != "opencode-go"
    assert escalated.get("_smart_router_escalated_from") == "medium"


def test_escalation_no_higher_tier_does_nothing():
    """When already at the highest tier, no escalation is possible."""
    gateway = FakeGateway()
    key = "telegram:1:2"

    # Route a message that actually gets complex score
    route_gateway_message(event("implementa un sistema distribuido con microservicios docker kubernetes CI/CD pipeline"), gateway)
    override = gateway._session_model_overrides[key]
    assert override["provider"] == "openai-codex"  # complex tier

    # Mark as errored
    override["_smart_router_error"] = True
    gateway._session_model_overrides[key] = override

    # Next message — must also be complex to stay in the same tier
    route_gateway_message(event("implementa un sistema de base de datos distribuido"), gateway)
    escalated = gateway._session_model_overrides[key]
    assert escalated["provider"] == "openai-codex"  # still complex, no higher tier


def test_check_and_escalate_public_api():
    gateway = FakeGateway()
    key = "telegram:1:2"

    # Route a message to create override
    route_gateway_message(event("explícame cómo funciona OAuth2"), gateway)
    override = gateway._session_model_overrides[key]
    override["_smart_router_error"] = True
    gateway._session_model_overrides[key] = override

    result = check_and_escalate_if_needed(gateway, key, "timeout")
    assert result is True
    escalated = gateway._session_model_overrides[key]
    assert escalated.get("_smart_router_escalated_from") == "medium"


def test_escalation_does_not_trigger_for_manual_overrides():
    """Only smart_router overrides should escalate, not manual ones."""
    gateway = FakeGateway()
    key = "telegram:1:2"
    gateway._session_model_overrides[key] = {"provider": "custom", "model": "manual"}

    result = check_and_escalate_if_needed(gateway, key, "error")
    assert result is False


# =============================================================================
# Runtime config persistence (Bug #3)
# =============================================================================

def test_runtime_config_persists_and_loads(monkeypatch, tmp_path):
    """Toggled values persist in the runtime config dict."""
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("hermes_smart_router.router._RUNTIME_STATE_PATH", state_file)

    gateway = FakeGateway()

    # Toggle dry-run on
    _toggle_config(gateway, "dry_run", True)
    cfg = _get_runtime_config(gateway)
    assert cfg.get("dry_run") is True

    # Toggle back
    _toggle_config(gateway, "dry_run", False)
    cfg = _get_runtime_config(gateway)
    assert cfg.get("dry_run") is False


def test_runtime_config_initialized_empty(monkeypatch, tmp_path):
    """First access initializes runtime config from empty state (no persisted file)."""
    state_file = tmp_path / "nonexistent.json"
    monkeypatch.setattr("hermes_smart_router.router._RUNTIME_STATE_PATH", state_file)

    gateway = FakeGateway()
    cfg = _get_runtime_config(gateway)
    assert isinstance(cfg, dict)
    assert len(cfg) == 0  # No persisted state exists


# =============================================================================
# Helpers
# =============================================================================

def load_config_for_test():
    from hermes_smart_router.config import (
        RouterConfig, ScoringConfig, PatternConfig, DEFAULT_ROUTES
    )
    return RouterConfig(
        enabled=True,
        dry_run=False,
        respect_manual_override=True,
        show_route_footer=True,
        llm_classifier_enabled=False,
        llm_classifier_provider="nous",
        llm_classifier_model="deepseek-v4-free",
        routes=list(DEFAULT_ROUTES),
        scoring=ScoringConfig(),
        patterns=PatternConfig(),
    )
