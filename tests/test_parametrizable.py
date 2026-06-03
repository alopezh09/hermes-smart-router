"""Tests for parametrizable routes with score ranges (LLM-only, no regex)."""

from hermes_smart_router.config import (
    RouterConfig,
    Route,
    DEFAULT_ROUTES,
    load_config,
)


def test_route_score_ranges():
    """Routes match by score range, not just by name."""
    routes = [
        Route(name="cheap", provider="nous", model="free",
              min_score=0, max_score=3, emoji="🟢"),
        Route(name="mid", provider="opencode", model="pro",
              min_score=4, max_score=7, emoji="🟡"),
        Route(name="high", provider="openai", model="gpt5",
              min_score=8, max_score=999, emoji="🔴"),
    ]
    cfg = RouterConfig(
        enabled=True, dry_run=False, respect_manual_override=True,
        show_route_footer=True,
        llm_classifier_provider="nous", llm_classifier_model="free",
        routes=routes,
    )
    assert cfg.route_for_score(0).name == "cheap"
    assert cfg.route_for_score(3).name == "cheap"
    assert cfg.route_for_score(4).name == "mid"
    assert cfg.route_for_score(8).name == "high"
    assert cfg.route_for_score(999).name == "high"


def test_negative_score_clamped():
    """Negative scores should be clamped to 0."""
    cfg = RouterConfig(
        enabled=True, dry_run=False, respect_manual_override=True,
        show_route_footer=True,
        llm_classifier_provider="nous", llm_classifier_model="free",
        routes=list(DEFAULT_ROUTES),
    )
    assert cfg.route_for_score(-5).name == "simple"


def test_route_for_backward_compat():
    """Legacy route_for(label) still works."""
    cfg = RouterConfig(
        enabled=True, dry_run=False, respect_manual_override=True,
        show_route_footer=True,
        llm_classifier_provider="nous", llm_classifier_model="free",
        routes=list(DEFAULT_ROUTES),
    )
    assert cfg.route_for("simple").provider == "nous"
    assert cfg.route_for("medium").provider == "opencode-go"
    assert cfg.route_for("complex").provider == "openai-codex"


def test_four_tier_routing():
    """User can define 4 custom tiers with their own score ranges."""
    routes = [
        Route(name="trivial", provider="nous", model="free",
              min_score=0, max_score=0, emoji="⚪"),
        Route(name="simple", provider="nous", model="free",
              min_score=1, max_score=2, emoji="🟢"),
        Route(name="medium", provider="opencode", model="pro",
              min_score=3, max_score=6, emoji="🟡"),
        Route(name="hard", provider="openai", model="gpt5",
              min_score=7, max_score=999, emoji="🔴"),
    ]
    cfg = RouterConfig(
        enabled=True, dry_run=False, respect_manual_override=True,
        show_route_footer=True,
        llm_classifier_provider="nous", llm_classifier_model="free",
        routes=routes,
    )
    assert cfg.route_for_score(0).name == "trivial"
    assert cfg.route_for_score(2).name == "simple"
    assert cfg.route_for_score(5).name == "medium"
    assert cfg.route_for_score(10).name == "hard"
    assert len(cfg.routes) == 4


def test_default_routes_model_strings():
    """Default routes use correct model strings (from user config, not flash:free)."""
    for route in DEFAULT_ROUTES:
        assert route.provider, f"Route {route.name} has no provider"
        assert route.model, f"Route {route.name} has no model"
        # The simple route should NOT have the flash:free suffix
        if route.name == "simple":
            assert "flash" not in route.model.lower(), \
                f"Simple route model '{route.model}' should not contain 'flash'"


def test_llm_classifier_config():
    """LLM classifier provider/model are configurable."""
    cfg = RouterConfig(
        enabled=True, dry_run=False, respect_manual_override=True,
        show_route_footer=True,
        llm_classifier_provider="openai-codex", llm_classifier_model="gpt-4o-mini",
        routes=list(DEFAULT_ROUTES),
    )
    assert cfg.llm_classifier_provider == "openai-codex"
    assert cfg.llm_classifier_model == "gpt-4o-mini"
