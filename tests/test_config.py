"""Tests for legacy config format auto-detection and new list format."""

from hermes_smart_router.config import load_config, Route


class FakeGateway:
    def __init__(self, config):
        self.config = config


def test_legacy_dict_format_still_works():
    """Old {simple: {provider, model}, medium: {...}, complex: {...}} format loads correctly."""
    gateway = FakeGateway({
        "smart_router": {
            "routes": {
                "simple": {"provider": "nous", "model": "free"},
                "medium": {"provider": "opencode", "model": "pro"},
                "complex": {"provider": "openai", "model": "gpt5"},
            }
        }
    })
    cfg = load_config(gateway)
    assert len(cfg.routes) == 3
    assert cfg.route_for("simple").provider == "nous"
    assert cfg.route_for("simple").model == "free"
    assert cfg.route_for("medium").provider == "opencode"
    assert cfg.route_for("complex").model == "gpt5"


def test_new_list_format():
    """New parametrizable list format loads correctly with score ranges."""
    gateway = FakeGateway({
        "smart_router": {
            "routes": [
                {"name": "quick", "emoji": "⚪", "min_score": 0, "max_score": 0,
                 "provider": "nous", "model": "free-fast"},
                {"name": "normal", "emoji": "🟡", "min_score": 1, "max_score": 5,
                 "provider": "opencode", "model": "pro"},
                {"name": "heavy", "emoji": "🔴", "min_score": 6, "max_score": 999,
                 "provider": "openai", "model": "gpt5"},
            ]
        }
    })
    cfg = load_config(gateway)
    assert len(cfg.routes) == 3
    assert cfg.route_for_score(0).name == "quick"
    assert cfg.route_for_score(0).provider == "nous"
    assert cfg.route_for_score(3).name == "normal"
    assert cfg.route_for_score(6).name == "heavy"


def test_llm_classifier_config_from_yaml():
    """LLM classifier provider and model are configurable from config.yaml."""
    gateway = FakeGateway({
        "smart_router": {
            "llm_classifier_provider": "openai-codex",
            "llm_classifier_model": "gpt-4o-mini",
        }
    })
    cfg = load_config(gateway)
    assert cfg.llm_classifier_provider == "openai-codex"
    assert cfg.llm_classifier_model == "gpt-4o-mini"


def test_legacy_format_with_new_keys():
    """Legacy dict format combined with new keys like emoji, min_score, max_score."""
    gateway = FakeGateway({
        "smart_router": {
            "routes": {
                "simple": {"provider": "nous", "model": "free", "emoji": "⚪", "min_score": 0, "max_score": 0},
                "medium": {"provider": "opencode", "model": "pro"},
                "complex": {"provider": "openai", "model": "gpt5"},
            }
        }
    })
    cfg = load_config(gateway)
    assert cfg.route_for_score(0).emoji == "⚪"
    assert cfg.route_for_score(0).name == "simple"


def test_no_config_returns_defaults():
    """No smart_router section at all returns sensible defaults."""
    gateway = FakeGateway({})
    cfg = load_config(gateway)
    assert cfg.enabled is True
    assert len(cfg.routes) == 3
    assert cfg.route_for_score(0).provider == "nous"
    # Default LLM classifier should be nous/openrouter/owl-alpha (free)
    assert cfg.llm_classifier_provider == "nous"
    assert cfg.llm_classifier_model == "openrouter/owl-alpha"


def test_route_names_property():
    """route_names helper returns all route names in order."""
    gateway = FakeGateway({
        "smart_router": {
            "routes": [
                {"name": "a", "provider": "p", "model": "m", "min_score": 0, "max_score": 1},
                {"name": "b", "provider": "p", "model": "m", "min_score": 2, "max_score": 5},
            ]
        }
    })
    cfg = load_config(gateway)
    assert cfg.route_names == ["a", "b"]
