"""Tests for router hook behavior (LLM classification mocked)."""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_smart_router.router import route_gateway_message
from hermes_smart_router.classifier import Classification


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


def _mock_simple():
    return Classification("simple", 0, "just a greeting")


def _mock_medium():
    return Classification("medium", 4, "explanation request")


def _mock_complex():
    return Classification("complex", 8, "implementation task")


@patch("hermes_smart_router.router.classify_message", return_value=_mock_complex())
def test_routes_complex_message(mock_classify):
    gateway = FakeGateway()
    result = route_gateway_message(event("Implementa un plugin con tests y GitHub Actions"), gateway)
    assert result == {"action": "allow"}
    key = "telegram:1:2"
    assert gateway._session_model_overrides[key]["provider"] == "openai-codex"
    assert gateway._session_model_overrides[key]["model"] == "gpt-5.5"
    assert gateway.evicted == [key]


@patch("hermes_smart_router.router.classify_message", return_value=_mock_complex())
def test_respects_manual_override_by_default(mock_classify):
    gateway = FakeGateway()
    key = "telegram:1:2"
    gateway._session_model_overrides[key] = {"provider": "custom", "model": "manual"}
    route_gateway_message(event("Implementa un plugin con tests"), gateway)
    assert gateway._session_model_overrides[key] == {"provider": "custom", "model": "manual"}


@patch("hermes_smart_router.router.classify_message", return_value=_mock_complex())
def test_dry_run_does_not_apply_override(mock_classify):
    gateway = FakeGateway({"smart_router": {"dry_run": True}})
    route_gateway_message(event("Implementa un plugin con tests"), gateway)
    assert gateway._session_model_overrides == {}
    assert gateway.evicted == []


def test_fails_open_without_gateway_internals():
    gateway = SimpleNamespace(config={})
    assert route_gateway_message(event("Implementa algo"), gateway) == {"action": "allow"}


@patch("hermes_smart_router.router.classify_message", return_value=_mock_medium())
def test_medium_route_populates_opencode_runtime_from_env(mock_classify, monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-opencode-key")
    gateway = FakeGateway()
    route_gateway_message(event("Explícame cómo funciona OAuth2"), gateway)
    override = gateway._session_model_overrides["telegram:1:2"]
    assert override["provider"] == "opencode-go"
    assert override["model"] == "deepseek-v4-pro"
    assert override["api_key"] == "test-opencode-key"
    assert override["base_url"] == "https://opencode.ai/zen/go/v1"
    assert override["api_mode"] == "chat_completions"


@patch("hermes_smart_router.router.classify_message", return_value=_mock_simple())
@patch("hermes_smart_router.classifier.classify_message", return_value=_mock_simple())
def test_classifier_command_classifies_free_text(mock_classify2, mock_classify1):
    gateway = FakeGateway()
    result = route_gateway_message(event("/smart-router classifier hola"), gateway)
    assert result["action"] == "rewrite"
    assert "simple" in result["text"]


@patch("hermes_smart_router.router.classify_message", return_value=_mock_complex())
def test_dry_run_command_classifies_free_text(mock_classify):
    gateway = FakeGateway()
    result = route_gateway_message(event("/smart-router dry-run Implementa un compilador desde cero"), gateway)
    assert result["action"] == "rewrite"
    assert "complex" in result["text"]


@patch("hermes_smart_router.router.classify_message", return_value=_mock_complex())
def test_custom_route_list_in_config(mock_classify):
    """New list format with custom route names and score ranges."""
    gateway = FakeGateway({
        "smart_router": {
            "routes": [
                {"name": "quick", "min_score": 0, "max_score": 2,
                 "provider": "nous", "model": "free-fast"},
                {"name": "heavy", "min_score": 3, "max_score": 999,
                 "provider": "openai", "model": "gpt5-custom"},
            ]
        }
    })
    result = route_gateway_message(event("Implementa un plugin con tests, CI/CD y deploy"), gateway)
    assert result == {"action": "allow"}
    override = gateway._session_model_overrides["telegram:1:2"]
    assert override["provider"] == "openai"
    assert override["model"] == "gpt5-custom"


@patch("hermes_smart_router.router.classify_message", return_value=_mock_simple())
def test_four_tier_router_config(mock_classify):
    """User defines 4 custom tiers and the router picks the right one."""
    gateway = FakeGateway({
        "smart_router": {
            "routes": [
                {"name": "trivial", "min_score": 0, "max_score": 0,
                 "emoji": "⚪", "provider": "nous", "model": "free"},
                {"name": "simple", "min_score": 1, "max_score": 2,
                 "emoji": "🟢", "provider": "nous", "model": "free"},
                {"name": "medium", "min_score": 3, "max_score": 6,
                 "emoji": "🟡", "provider": "opencode", "model": "pro"},
                {"name": "hard", "min_score": 7, "max_score": 999,
                 "emoji": "🔴", "provider": "openai", "model": "gpt5"},
            ]
        }
    })
    route_gateway_message(event("gracias"), gateway)
    override = gateway._session_model_overrides["telegram:1:2"]
    assert override["provider"] == "nous"
    assert override["_smart_router_route"] == "trivial"


@patch("hermes_smart_router.router.classify_message", return_value=None)
def test_llm_failure_fails_open(mock_classify):
    """When LLM classifier returns None, router allows default model."""
    gateway = FakeGateway()
    result = route_gateway_message(event("Hola"), gateway)
    assert result == {"action": "allow"}
    assert gateway._session_model_overrides == {}
