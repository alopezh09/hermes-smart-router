"""Additional tests for commands.py coverage and config edge cases."""

import copy
from types import SimpleNamespace

from hermes_smart_router.commands import (
    handle_slash_command,
    format_help,
)
from hermes_smart_router.config import (
    RouterConfig,
    ScoringConfig,
    PatternConfig,
    DEFAULT_ROUTES,
    _load_routes_from_list,
    load_config,
)


def _make_cfg(**overrides):
    """Build a RouterConfig with defaults, optionally overridden."""
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
        **overrides,
    )


class FakeGateway:
    def __init__(self, config=None):
        self.config = config or {}
        self._session_model_overrides = {}

    def _session_key_for_source(self, source):
        return f"{source.platform}:{source.chat_id}:{source.user_id}"

    def _evict_cached_agent(self, session_key):
        pass


def test_commands_help_output():
    output = format_help()
    assert "Smart Router Commands" in output
    assert "test" in output
    assert "metrics" in output
    assert "version" in output
    assert "wizard" in output


def test_commands_status_returns_routes_table():
    gateway = FakeGateway({"smart_router": {"enabled": True}})
    source = SimpleNamespace(platform="telegram", chat_id="1")
    cfg = _make_cfg()
    result = handle_slash_command("/smart-router", gateway, source, cfg)
    assert result is not None
    assert "Routes" in result or "route" in result.lower()


def test_commands_routes_subcommand():
    gateway = FakeGateway()
    source = SimpleNamespace(platform="telegram", chat_id="1")
    cfg = _make_cfg()
    result = handle_slash_command("/smart-router routes", gateway, source, cfg)
    assert result is not None


def test_commands_weights_subcommand():
    gateway = FakeGateway()
    source = SimpleNamespace(platform="telegram", chat_id="1")
    cfg = _make_cfg()
    result = handle_slash_command("/smart-router weights", gateway, source, cfg)
    assert result is not None
    assert "Scoring Weights" in result


def test_commands_patterns_subcommand():
    gateway = FakeGateway()
    source = SimpleNamespace(platform="telegram", chat_id="1")
    cfg = _make_cfg()
    result = handle_slash_command("/smart-router patterns", gateway, source, cfg)
    assert result is not None
    assert "COMPLEX" in result or "complex" in result


def test_commands_wizard_subcommand():
    gateway = FakeGateway()
    source = SimpleNamespace(platform="telegram", chat_id="1")
    cfg = _make_cfg()
    result = handle_slash_command("/smart-router wizard", gateway, source, cfg)
    assert result is not None
    assert "Setup Wizard" in result


def test_commands_help_subcommand():
    gateway = FakeGateway()
    source = SimpleNamespace(platform="telegram", chat_id="1")
    cfg = _make_cfg()
    result = handle_slash_command("/smart-router help", gateway, source, cfg)
    assert result is not None
    assert "Smart Router Commands" in result


def test_commands_test_returns_none_for_router_handling():
    """test subcommand returns None so router.py can handle it."""
    gateway = FakeGateway()
    source = SimpleNamespace(platform="telegram", chat_id="1")
    cfg = _make_cfg()
    result = handle_slash_command("/smart-router test implementa algo", gateway, source, cfg)
    assert result is None  # Passes through to router.py


def test_commands_metrics_returns_none_for_router_handling():
    gateway = FakeGateway()
    source = SimpleNamespace(platform="telegram", chat_id="1")
    cfg = _make_cfg()
    result = handle_slash_command("/smart-router metrics", gateway, source, cfg)
    assert result is None


def test_commands_version_returns_none_for_router_handling():
    gateway = FakeGateway()
    source = SimpleNamespace(platform="telegram", chat_id="1")
    cfg = _make_cfg()
    result = handle_slash_command("/smart-router version", gateway, source, cfg)
    assert result is None


def test_commands_unknown_subcommand_returns_none():
    gateway = FakeGateway()
    source = SimpleNamespace(platform="telegram", chat_id="1")
    cfg = _make_cfg()
    result = handle_slash_command("/smart-router nonexistent", gateway, source, cfg)
    assert result is None


# =============================================================================
# Config edge cases
# =============================================================================

def test_overlapping_score_ranges_first_wins():
    """When two routes overlap, the first one that matches wins."""
    routes = _load_routes_from_list([
        {"name": "a", "min_score": 0, "max_score": 5,
         "provider": "p1", "model": "m1"},
        {"name": "b", "min_score": 3, "max_score": 999,
         "provider": "p2", "model": "m2"},
    ])
    cfg = RouterConfig(
        enabled=True, dry_run=False, respect_manual_override=True,
        show_route_footer=True, llm_classifier_enabled=False,
        llm_classifier_provider="nous", llm_classifier_model="free",
        routes=routes, scoring=ScoringConfig(), patterns=PatternConfig(),
    )

    # Score 4 should match route "a" even though it also overlaps with "b"
    r = cfg.route_for_score(4)
    assert r.name == "a"


def test_score_below_minimum_clamped_to_zero():
    """Negative scores are clamped to 0, matching the lowest tier."""
    cfg = _make_cfg()
    r = cfg.route_for_score(-10)
    assert r.min_score <= 0


def test_legacy_dict_preserves_emoji():
    """Legacy config format routes get the correct emoji."""
    raw = {
        "simple": {"provider": "nous", "model": "free"},
        "medium": {"provider": "opencode", "model": "pro"},
        "complex": {"provider": "openai", "model": "gpt5"},
    }
    from hermes_smart_router.config import _load_routes
    routes = _load_routes(raw)
    assert routes[0].emoji == "🟢"   # simple
    assert routes[1].emoji == "🟡"   # medium
    assert routes[2].emoji == "🔴"   # complex


def test_load_config_with_mixed_config():
    """load_config handles a realistic gateway config shape."""
    gateway = FakeGateway({
        "smart_router": {
            "enabled": True,
            "show_route_footer": True,
            "llm_classifier_enabled": False,
            "routes": {
                "simple": {"provider": "nous", "model": "free"},
                "medium": {"provider": "opencode-go", "model": "pro"},
                "complex": {"provider": "openai-codex", "model": "gpt5"},
            },
        }
    })
    cfg = load_config(gateway)
    assert cfg.enabled is True
    assert len(cfg.routes) == 3
    assert cfg.routes[0].name == "simple"
    assert cfg.routes[1].name == "medium"
    assert cfg.routes[2].name == "complex"


def test_empty_routes_falls_back_to_defaults():
    """When routes config is empty, defaults are used."""
    from hermes_smart_router.config import _load_routes
    routes = _load_routes({})
    assert len(routes) == 3  # simple, medium, complex defaults
