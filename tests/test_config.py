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
    # Score 0 → quick (first match)
    assert cfg.route_for_score(0).name == "quick"
    assert cfg.route_for_score(0).provider == "nous"
    # Score 3 → normal
    assert cfg.route_for_score(3).name == "normal"
    # Score 6 → heavy
    assert cfg.route_for_score(6).name == "heavy"


def test_custom_scoring_weights_in_config():
    """Custom scoring weights loaded from config."""
    gateway = FakeGateway({
        "smart_router": {
            "scoring": {
                "weight_complex_pattern": 10,
                "weight_medium_pattern": 5,
                "complex_threshold": 10,
                "medium_threshold": 4,
            }
        }
    })
    cfg = load_config(gateway)
    assert cfg.scoring.weight_complex_pattern == 10
    assert cfg.scoring.weight_medium_pattern == 5
    assert cfg.scoring.complex_threshold == 10
    assert cfg.scoring.medium_threshold == 4


def test_custom_patterns_in_config():
    """Custom regex patterns loaded from config."""
    gateway = FakeGateway({
        "smart_router": {
            "patterns": {
                "complex": ["urgente", "crítico"],
                "medium": ["consulta"],
                "simple": ["ok", "gracias"],
            }
        }
    })
    cfg = load_config(gateway)
    assert "urgente" in cfg.patterns.complex
    assert "consulta" in cfg.patterns.medium
    assert "ok" in cfg.patterns.simple


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
