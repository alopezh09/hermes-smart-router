from types import SimpleNamespace

from hermes_smart_router.router import route_gateway_message


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


def test_routes_complex_message():
    gateway = FakeGateway()
    result = route_gateway_message(event("Implementa un plugin con tests y GitHub Actions"), gateway)
    assert result == {"action": "allow"}
    key = "telegram:1:2"
    assert gateway._session_model_overrides[key]["provider"] == "openai-codex"
    assert gateway._session_model_overrides[key]["model"] == "gpt-5.5"
    assert gateway.evicted == [key]


def test_respects_manual_override_by_default():
    gateway = FakeGateway()
    key = "telegram:1:2"
    gateway._session_model_overrides[key] = {"provider": "custom", "model": "manual"}
    route_gateway_message(event("Implementa un plugin con tests"), gateway)
    assert gateway._session_model_overrides[key] == {"provider": "custom", "model": "manual"}


def test_dry_run_does_not_apply_override():
    gateway = FakeGateway({"smart_router": {"dry_run": True}})
    route_gateway_message(event("Implementa un plugin con tests"), gateway)
    assert gateway._session_model_overrides == {}
    assert gateway.evicted == []


def test_fails_open_without_gateway_internals():
    gateway = SimpleNamespace(config={})
    assert route_gateway_message(event("Implementa algo"), gateway) == {"action": "allow"}


def test_medium_route_populates_opencode_runtime_from_env(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-opencode-key")
    gateway = FakeGateway()
    route_gateway_message(event("Explícame cómo funciona OAuth2"), gateway)
    override = gateway._session_model_overrides["telegram:1:2"]
    assert override["provider"] == "opencode-go"
    assert override["model"] == "deepseek-v4-pro"
    assert override["api_key"] == "test-opencode-key"
    assert override["base_url"] == "https://opencode.ai/zen/go/v1"
    assert override["api_mode"] == "chat_completions"


def test_classifier_command_classifies_free_text_without_reaching_agent():
    gateway = FakeGateway()
    result = route_gateway_message(event("/smart-router classifier hola"), gateway)
    assert result["action"] == "rewrite"
    assert "simple" in result["text"]
    assert "nous/deepseek-v4-free" in result["text"]


def test_dry_run_command_classifies_free_text_without_toggling_mode():
    gateway = FakeGateway()
    result = route_gateway_message(event("/smart-router dry-run escribe un compilador desde cero"), gateway)
    assert result["action"] == "rewrite"
    assert "complex" in result["text"]
    assert "openai-codex/gpt-5.5" in result["text"]
    assert not hasattr(gateway, "_smart_router_runtime_config")
