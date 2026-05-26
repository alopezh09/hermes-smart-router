"""Configuration loading for Hermes Smart Router.

Parametrizable routes with score ranges, configurable scoring weights,
and user-defined regex patterns — all driven from ``hermes config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Route:
    """A concrete provider/model route to apply to a Hermes gateway session."""

    name: str
    provider: str
    model: str
    min_score: int = 0
    max_score: int = 999
    emoji: str = "⚪"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_mode: Optional[str] = None

    def as_override(self) -> Dict[str, Any]:
        """Return the shape Hermes gateway session model overrides expect."""
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "_smart_router": True,
            "_smart_router_route": self.name,
        }

    def matches_score(self, score: int) -> bool:
        """Check whether *score* falls within this route's range (inclusive)."""
        return self.min_score <= score <= self.max_score


@dataclass(frozen=True)
class ScoringConfig:
    """Weights used by the regex classifier to compute a complexity score."""

    weight_complex_pattern: int = 4
    weight_medium_pattern: int = 2
    weight_simple_pattern: int = -3
    weight_code_block: int = 5
    weight_very_long: int = 3       # >= 80 words
    weight_long: int = 2            # >= 35 words
    weight_requirement_list: int = 1  # per bullet / numbered / list marker (max 3)

    # Score thresholds are now per-route (min_score / max_score), so
    # these two are kept only for backward-compatible label mapping when
    # a caller still uses the old "complexity label" API.
    complex_threshold: int = 6
    medium_threshold: int = 2


@dataclass(frozen=True)
class PatternConfig:
    """Regex patterns for each complexity tier, fully user-configurable.

    Each value is a list of regex strings (compiled with ``re.IGNORECASE``
    and ``re.search`` at classification time).
    """

    complex: List[str] = field(default_factory=lambda: _DEFAULT_COMPLEX_PATTERNS[:])
    medium: List[str] = field(default_factory=lambda: _DEFAULT_MEDIUM_PATTERNS[:])
    simple: List[str] = field(default_factory=lambda: _DEFAULT_SIMPLE_PATTERNS[:])


@dataclass(frozen=True)
class RouterConfig:
    enabled: bool
    dry_run: bool
    respect_manual_override: bool
    show_route_footer: bool
    llm_classifier_enabled: bool
    llm_classifier_provider: str
    llm_classifier_model: str
    routes: List[Route]            # ordered by score range, ascending
    scoring: ScoringConfig
    patterns: PatternConfig

    def route_for_score(self, score: int) -> Route:
        """Pick the first route whose score range contains *score*.

        Negative scores are clamped to 0 so they always match the
        lowest available tier.
        """
        effective = max(0, score)
        for route in self.routes:
            if route.matches_score(effective):
                return route
        return self.routes[-1]  # fallback: last/highest route

    def route_for(self, complexity: str) -> Route:
        """Backward-compatible lookup by complexity label.

        First tries a by-name match on route names (``simple``, ``medium``,
        ``complex``), then falls back to score-based mapping using the legacy
        *complex_threshold* / *medium_threshold* from :class:`ScoringConfig`.
        """
        for route in self.routes:
            if route.name.lower() == complexity.lower():
                return route
        # Score-based fallback for bare labels
        if complexity == "complex":
            return self.route_for_score(self.scoring.complex_threshold)
        if complexity == "medium":
            return self.route_for_score(self.scoring.medium_threshold)
        return self.route_for_score(0)

    @property
    def route_names(self) -> List[str]:
        return [r.name for r in self.routes]


# ---------------------------------------------------------------------------
# Default patterns (mirrors the current hardcoded lists)
# ---------------------------------------------------------------------------

_DEFAULT_COMPLEX_PATTERNS = [
    r"\bimplement(a|ar|ación|ation)?\b",
    r"\brefactor\b",
    r"\bdebug\b",
    r"\bbug\b",
    r"\btest(s|ing)?\b",
    r"\bdeploy\b",
    r"\bproduction\b",
    r"\barchitecture\b",
    r"\barquitectura\b",
    r"\bplugin\b",
    r"\bgithub\b",
    r"\bpull request\b|\bPR\b",
    r"\bcodebase\b",
    r"\brepo(sitory)?\b",
    r"\bbase de código\b",
    r"\bend[- ]?to[- ]?end\b",
    r"\bcompleto\b",
    r"\bautomatiz(a|ar|ación)\b",
    r"\bcompilador\b",
    r"\bcompiler\b",
    r"\bdesde cero\b",
    r"\bfrom scratch\b",
]

_DEFAULT_MEDIUM_PATTERNS = [
    r"\bexplica(r|me)?\b",
    r"\bexplícame\b",
    r"\bresume(n|ir)?\b",
    r"\bcompara(r)?\b",
    r"\bplan\b",
    r"\bdiseña(r)?\b",
    r"\banaliza(r)?\b",
    r"\bescribe\b",
    r"\bdraft\b",
    r"\bmejora(r)?\b",
]

_DEFAULT_SIMPLE_PATTERNS = [
    r"^(ok|dale|gracias|perfecto|sí|si|no|listo|va|claro)[!.\s]*$",
    r"\bqué hora\b",
    r"\bcuánto es\b",
    r"\bdefine\b",
]

# ---------------------------------------------------------------------------
# Default routes (ordered by score range)
# ---------------------------------------------------------------------------

DEFAULT_ROUTES: List[Route] = [
    Route(name="simple", provider="nous", model="deepseek/deepseek-v4-flash:free",
          min_score=0, max_score=1, emoji="🟢"),
    Route(name="medium", provider="opencode-go", model="deepseek-v4-pro",
          min_score=2, max_score=5, emoji="🟡"),
    Route(name="complex", provider="openai-codex", model="gpt-5.5",
          min_score=6, max_score=999, emoji="🔴"),
]

DEFAULT_SCORING = ScoringConfig()
DEFAULT_PATTERNS = PatternConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    data = {}
    for key in ("enabled", "dry_run", "respect_manual_override",
                "show_route_footer", "llm_classifier_enabled",
                "llm_classifier_provider", "llm_classifier_model",
                "routes", "scoring", "patterns"):
        if hasattr(value, key):
            data[key] = getattr(value, key)
    return data


def _get_smart_router_section(gateway: Any) -> Mapping[str, Any]:
    config = getattr(gateway, "config", None)
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return _to_mapping(config.get("smart_router"))
    return _to_mapping(getattr(config, "smart_router", None))


# ---------------------------------------------------------------------------
# Route loading (handles both legacy dict format and new list format)
# ---------------------------------------------------------------------------

def _load_routes(raw_routes: Any) -> List[Route]:
    """Load routes from config, handling both legacy dict and new list format.

    Legacy format (detected automatically):
        simple: {provider: nous, model: deepseek-v4-free}
        medium: {provider: opencode-go, model: deepseek-v4-pro}
        complex: {provider: openai-codex, model: gpt-5.5}

    New format:
        - name: cheap
          min_score: 0
          max_score: 3
          provider: nous
          model: deepseek-v4-free
    """
    if isinstance(raw_routes, list):
        return _load_routes_from_list(raw_routes)
    if isinstance(raw_routes, Mapping) and raw_routes:
        return _load_routes_from_legacy_dict(raw_routes)
    return list(DEFAULT_ROUTES)


def _load_routes_from_list(routes_list: list) -> List[Route]:
    """Load routes from new-style list format."""
    loaded: List[Route] = []
    for idx, entry in enumerate(routes_list):
        data = _to_mapping(entry)
        if not data:
            continue
        name = str(data.get("name") or f"route_{idx}")
        loaded.append(Route(
            name=name,
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            min_score=int(data.get("min_score", 0)),
            max_score=int(data.get("max_score", 999)),
            emoji=str(data.get("emoji", "⚪")),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            api_mode=data.get("api_mode"),
        ))
    if not loaded:
        return list(DEFAULT_ROUTES)
    # Sort by min_score ascending
    loaded.sort(key=lambda r: r.min_score)
    return loaded


def _load_routes_from_legacy_dict(raw: Mapping) -> List[Route]:
    """Convert legacy ``{simple: {...}, medium: {...}, complex: {...}}`` format.

    Maps complexity labels to score ranges using default thresholds:
      simple  -> 0-1
      medium  -> 2-5
      complex -> 6-999

    Users can override those ranges by including min_score/max_score keys
    inside each legacy entry.
    """
    _LEGACY_DEFAULTS = {
        "simple":  {"min_score": 0,  "max_score": 1, "emoji": "🟢"},
        "medium":  {"min_score": 2,  "max_score": 5, "emoji": "🟡"},
        "complex": {"min_score": 6,  "max_score": 999, "emoji": "🔴"},
    }
    routes: List[Route] = []
    for name in ("simple", "medium", "complex"):
        entry = _to_mapping(raw.get(name))
        if not entry:
            legacy = _LEGACY_DEFAULTS[name]
            fallback = DEFAULT_ROUTES[{"simple": 0, "medium": 1, "complex": 2}[name]]
            routes.append(Route(
                name=name,
                provider=fallback.provider,
                model=fallback.model,
                min_score=int(entry.get("min_score", legacy["min_score"])),
                max_score=int(entry.get("max_score", legacy["max_score"])),
                emoji=str(entry.get("emoji", legacy["emoji"])),
            ))
            continue
        legacy = _LEGACY_DEFAULTS[name]
        fallback = DEFAULT_ROUTES[{"simple": 0, "medium": 1, "complex": 2}[name]]
        routes.append(Route(
            name=name,
            provider=str(entry.get("provider") or fallback.provider),
            model=str(entry.get("model") or fallback.model),
            min_score=int(entry.get("min_score", legacy["min_score"])),
            max_score=int(entry.get("max_score", legacy["max_score"])),
            emoji=str(entry.get("emoji", legacy["emoji"])),
            api_key=entry.get("api_key"),
            base_url=entry.get("base_url"),
            api_mode=entry.get("api_mode"),
        ))
    return routes


# ---------------------------------------------------------------------------
# Scoring / patterns loading
# ---------------------------------------------------------------------------

def _load_scoring(raw: Any) -> ScoringConfig:
    data = _to_mapping(raw)
    if not data:
        return ScoringConfig()
    return ScoringConfig(
        weight_complex_pattern=int(data.get("weight_complex_pattern", 4)),
        weight_medium_pattern=int(data.get("weight_medium_pattern", 2)),
        weight_simple_pattern=int(data.get("weight_simple_pattern", -3)),
        weight_code_block=int(data.get("weight_code_block", 5)),
        weight_very_long=int(data.get("weight_very_long", 3)),
        weight_long=int(data.get("weight_long", 2)),
        weight_requirement_list=int(data.get("weight_requirement_list", 1)),
        complex_threshold=int(data.get("complex_threshold", 6)),
        medium_threshold=int(data.get("medium_threshold", 2)),
    )


def _load_patterns(raw: Any) -> PatternConfig:
    data = _to_mapping(raw)
    if not data:
        return PatternConfig()
    return PatternConfig(
        complex=[str(p) for p in data.get("complex", [])] or _DEFAULT_COMPLEX_PATTERNS[:],
        medium=[str(p) for p in data.get("medium", [])] or _DEFAULT_MEDIUM_PATTERNS[:],
        simple=[str(p) for p in data.get("simple", [])] or _DEFAULT_SIMPLE_PATTERNS[:],
    )


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def load_config(gateway: Any = None) -> RouterConfig:
    """Load plugin config from ``gateway.config.smart_router`` if present.

    Supports both the legacy dict ``routes: {simple, medium, complex}`` format
    and the new parametrizable list format with score ranges, custom scoring
    weights, and user-defined regex patterns.
    """
    section = _get_smart_router_section(gateway)
    raw_routes = section.get("routes", {})
    raw_scoring = section.get("scoring", {})
    raw_patterns = section.get("patterns", {})

    return RouterConfig(
        enabled=bool(section.get("enabled", True)),
        dry_run=bool(section.get("dry_run", False)),
        respect_manual_override=bool(section.get("respect_manual_override", True)),
        show_route_footer=bool(section.get("show_route_footer", True)),
        llm_classifier_enabled=bool(section.get("llm_classifier_enabled", False)),
        llm_classifier_provider=str(section.get("llm_classifier_provider", "nous")),
        llm_classifier_model=str(section.get("llm_classifier_model", "deepseek/deepseek-v4-flash:free")),
        routes=_load_routes(raw_routes),
        scoring=_load_scoring(raw_scoring),
        patterns=_load_patterns(raw_patterns),
    )
