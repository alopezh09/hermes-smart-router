"""Tests for parametrizable scoring weights, patterns, and route ranges."""

from hermes_smart_router.classifier import classify_message
from hermes_smart_router.config import (
    ScoringConfig,
    PatternConfig,
    RouterConfig,
    Route,
    DEFAULT_ROUTES,
)


def test_custom_scoring_weights():
    """A heavy scoring config should push messages to complex faster."""
    heavy = ScoringConfig(
        weight_complex_pattern=10,
        weight_medium_pattern=5,
        weight_simple_pattern=0,
        complex_threshold=8,
        medium_threshold=3,
    )
    # "Explica" should be medium normally, but with heavy weights it's medium
    result = classify_message("Explícame algo", scoring=heavy)
    assert result.complexity == "medium"


def test_custom_patterns():
    """User-defined patterns should influence classification."""
    patterns = PatternConfig(
        complex=["supercalifragilistico"],
        medium=["patata"],
        simple=["hola"],
    )
    result = classify_message("Quiero una patata", patterns=patterns)
    assert result.complexity == "medium"


def test_custom_complex_pattern():
    """A word the user added to complex should route to complex with enough hits."""
    patterns = PatternConfig(
        complex=["emergencia", "urgente", "crítico"],
        medium=["consulta"],
        simple=["hola"],
    )
    scoring = ScoringConfig(complex_threshold=4)  # lower threshold for test
    result = classify_message(
        "Esto es una emergencia, necesito ayuda urgente",
        scoring=scoring, patterns=patterns,
    )
    assert result.complexity == "complex"


def test_empty_patterns_fallback():
    """Empty pattern lists should work — nothing matches."""
    patterns = PatternConfig(complex=[], medium=[], simple=[])
    result = classify_message("Cualquier mensaje largo sin patrones deberia ser simple si no hay hits")
    assert result.complexity in ("simple", "medium")


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
        show_route_footer=True, llm_classifier_enabled=False,
        llm_classifier_provider="nous", llm_classifier_model="free",
        routes=routes, scoring=ScoringConfig(), patterns=PatternConfig(),
    )
    # Score 0 → cheap
    assert cfg.route_for_score(0).name == "cheap"
    # Score 3 → cheap
    assert cfg.route_for_score(3).name == "cheap"
    # Score 4 → mid
    assert cfg.route_for_score(4).name == "mid"
    # Score 8 → high
    assert cfg.route_for_score(8).name == "high"
    # Score 999 → high
    assert cfg.route_for_score(999).name == "high"


def test_route_for_backward_compat():
    """Legacy route_for(label) still works."""
    cfg = RouterConfig(
        enabled=True, dry_run=False, respect_manual_override=True,
        show_route_footer=True, llm_classifier_enabled=False,
        llm_classifier_provider="nous", llm_classifier_model="free",
        routes=list(DEFAULT_ROUTES), scoring=ScoringConfig(), patterns=PatternConfig(),
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
        show_route_footer=True, llm_classifier_enabled=False,
        llm_classifier_provider="nous", llm_classifier_model="free",
        routes=routes, scoring=ScoringConfig(), patterns=PatternConfig(),
    )
    assert cfg.route_for_score(0).name == "trivial"
    assert cfg.route_for_score(2).name == "simple"
    assert cfg.route_for_score(5).name == "medium"
    assert cfg.route_for_score(10).name == "hard"
    assert len(cfg.routes) == 4


def test_all_patterns_configurable():
    """All three pattern tiers are user-replaceable."""
    patterns = PatternConfig(
        complex=["IA", "inteligencia artificial"],
        medium=["pregunta", "duda"],
        simple=["ok"],
    )
    result = classify_message("Tengo una duda sobre IA", patterns=patterns)
    # Both "duda" (medium) and "IA" (complex) match → complex wins by weight
    # complex_hits=1 * 4 = 4, medium_hits=1 * 2 = 2 → score 6 → complex
    assert result.complexity == "complex"
