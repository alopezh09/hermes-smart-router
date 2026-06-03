"""Tests for commands.py — slash command handlers, visual tables, and wizard."""

import pytest
from hermes_smart_router.config import load_config, RouterConfig, Route
from hermes_smart_router.commands import (
    format_routes_table,
    discover_hermes_models,
    format_models_table,
    generate_wizard_guide,
    handle_slash_command,
    format_help,
)


class FakeGateway:
    def __init__(self, config=None):
        self.config = config or {}


class FakeSource:
    def __init__(self, platform="telegram", chat_id="123"):
        self.platform = platform
        self.chat_id = chat_id


def _make_cfg(routes=None, **kwargs):
    """Helper: build a RouterConfig with defaults."""
    return load_config(FakeGateway({
        "smart_router": dict(
            routes=routes or [
                {"name": "simple", "min_score": 0, "max_score": 1,
                 "provider": "nous", "model": "deepseek-v4-free", "emoji": "🟢"},
                {"name": "medium", "min_score": 2, "max_score": 5,
                 "provider": "opencode-go", "model": "deepseek-v4-pro", "emoji": "🟡"},
                {"name": "complex", "min_score": 6, "max_score": 999,
                 "provider": "openai-codex", "model": "gpt-5.5", "emoji": "🔴"},
            ],
            **kwargs,
        ),
    }))


class TestFormatRoutesTable:
    def test_returns_string_with_routes(self):
        cfg = _make_cfg()
        result = format_routes_table(cfg)
        assert isinstance(result, str)
        assert "simple" in result
        assert "medium" in result
        assert "complex" in result
        assert "nous" in result
        assert "openai-codex" in result

    def test_includes_emoji(self):
        cfg = _make_cfg()
        result = format_routes_table(cfg)
        assert "🟢" in result
        assert "🟡" in result
        assert "🔴" in result

    def test_includes_score_ranges(self):
        cfg = _make_cfg()
        result = format_routes_table(cfg)
        assert "0–1" in result or "0-1" in result
        assert "999" in result

    def test_shows_total_count(self):
        cfg = _make_cfg()
        result = format_routes_table(cfg)
        assert "Total routes" in result

    def test_shows_llm_classifier_info(self):
        cfg = _make_cfg(llm_classifier_provider="nous", llm_classifier_model="deepseek-v4-free")
        result = format_routes_table(cfg)
        assert "LLM" in result
        assert "nous" in result

    def test_shows_dry_run_status(self):
        cfg = _make_cfg(dry_run=True)
        result = format_routes_table(cfg)
        assert "ON" in result or "🔬" in result

    def test_empty_routes(self):
        cfg = _make_cfg(routes=[])
        result = format_routes_table(cfg)
        assert isinstance(result, str)
        assert "**0**" in result or "0" in result


class TestDiscoverHermesModels:
    def test_discovers_primary_model(self):
        gateway = FakeGateway({
            "model": {"provider": "openai", "default": "gpt-4"},
        })
        models = discover_hermes_models(gateway)
        assert any(m["provider"] == "openai" and m["model"] == "gpt-4" for m in models)

    def test_primary_tagged_as_primary(self):
        gateway = FakeGateway({
            "model": {"provider": "openai", "default": "gpt-4"},
        })
        models = discover_hermes_models(gateway)
        primary = [m for m in models if m["provider"] == "openai"]
        assert len(primary) == 1
        assert primary[0]["source"] == "primary"

    def test_discovers_fallback_models(self):
        gateway = FakeGateway({
            "model": {"provider": "openai", "default": "gpt-4"},
            "fallback_providers": [
                {"provider": "nous", "model": "deepseek"},
                {"provider": "anthropic", "model": "claude"},
            ],
        })
        models = discover_hermes_models(gateway)
        assert any(m["provider"] == "nous" and m["model"] == "deepseek" for m in models)
        assert any(m["provider"] == "anthropic" and m["model"] == "claude" for m in models)

    def test_fallbacks_tagged_as_fallback(self):
        gateway = FakeGateway({
            "fallback_providers": [
                {"provider": "nous", "model": "deepseek"},
            ],
        })
        models = discover_hermes_models(gateway)
        fb = [m for m in models if m["provider"] == "nous"]
        assert len(fb) == 1
        assert fb[0]["source"] == "fallback"

    def test_no_duplicates(self):
        gateway = FakeGateway({
            "model": {"provider": "nous", "default": "deepseek"},
            "fallback_providers": [
                {"provider": "nous", "model": "deepseek"},
            ],
        })
        models = discover_hermes_models(gateway)
        nous_entries = [m for m in models if m["provider"] == "nous"]
        assert len(nous_entries) == 1

    def test_empty_config_returns_empty(self):
        gateway = FakeGateway({})
        models = discover_hermes_models(gateway)
        assert models == []

    def test_null_gateway(self):
        models = discover_hermes_models(None)
        assert models == []


class TestFormatModelsTable:
    def test_returns_table_with_discovered_models(self):
        gateway = FakeGateway({
            "model": {"provider": "openai", "default": "gpt-4"},
            "fallback_providers": [
                {"provider": "nous", "model": "deepseek"},
            ],
        })
        result = format_models_table(gateway)
        assert "openai" in result
        assert "gpt-4" in result
        assert "nous" in result
        assert "deepseek" in result

    def test_no_models_shows_message(self):
        gateway = FakeGateway({})
        result = format_models_table(gateway)
        assert "No models detected" in result or "Total: **0**" in result


class TestWizard:
    def test_returns_string_with_steps(self):
        gateway = FakeGateway({
            "model": {"provider": "openai", "default": "gpt-4"},
        })
        result = generate_wizard_guide(gateway)
        assert isinstance(result, str)
        assert "smart_router" in result

    def test_includes_config_yaml_snippet(self):
        gateway = FakeGateway({
            "model": {"provider": "openai", "default": "gpt-4"},
        })
        result = generate_wizard_guide(gateway)
        assert "yaml" in result.lower() or "provider:" in result
        assert "routes:" in result

    def test_includes_llm_info(self):
        gateway = FakeGateway({
            "smart_router": {
                "llm_classifier_provider": "nous",
                "llm_classifier_model": "deepseek-v4-free",
            }
        })
        result = generate_wizard_guide(gateway)
        assert "LLM" in result
        assert "nous" in result


class TestHandleSlashCommand:
    def _gw(self):
        return FakeGateway({
            "model": {"provider": "openai", "default": "gpt-4"},
        })

    def _src(self):
        return FakeSource()

    def _cfg(self):
        return _make_cfg()

    def test_status_returns_table(self):
        result = handle_slash_command("/smart-router", self._gw(), self._src(), self._cfg())
        assert result is not None
        assert "simple" in result

    def test_routes_returns_table(self):
        result = handle_slash_command("/smart-router routes", self._gw(), self._src(), self._cfg())
        assert result is not None
        assert "Routes" in result

    def test_models_returns_table(self):
        result = handle_slash_command("/smart-router models", self._gw(), self._src(), self._cfg())
        assert result is not None
        assert "gpt-4" in result

    def test_wizard_returns_guide(self):
        result = handle_slash_command("/smart-router wizard", self._gw(), self._src(), self._cfg())
        assert result is not None
        assert "smart_router" in result.lower()

    def test_help_returns_help_text(self):
        result = handle_slash_command("/smart-router help", self._gw(), self._src(), self._cfg())
        assert result is not None
        assert "/smart-router" in result

    def test_add_snippet_returns_yaml(self):
        result = handle_slash_command(
            "/smart-router routes add-snippet premium 6 999 openai-codex gpt-5.5",
            self._gw(), self._src(), self._cfg(),
        )
        assert result is not None
        assert "premium" in result
        assert "openai-codex" in result

    def test_add_snippet_missing_args_shows_usage(self):
        result = handle_slash_command(
            "/smart-router routes add-snippet foo",
            self._gw(), self._src(), self._cfg(),
        )
        assert result is not None
        assert "Usage" in result

    def test_remove_snippet_shows_instructions(self):
        result = handle_slash_command(
            "/smart-router routes remove-snippet simple",
            self._gw(), self._src(), self._cfg(),
        )
        assert result is not None
        assert "Remove" in result or "delete" in result.lower()

    def test_non_smart_router_command_returns_none(self):
        result = handle_slash_command("/help", self._gw(), self._src(), self._cfg())
        assert result is None


class TestFormatHelp:
    def test_includes_all_sections(self):
        result = format_help()
        assert "routes" in result.lower()
        assert "models" in result.lower()
        assert "wizard" in result.lower()
        assert "LLM" in result
