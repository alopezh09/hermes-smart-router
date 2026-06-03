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

    weight_complex_pattern: int = 6
    weight_medium_pattern: int = 2
    weight_simple_pattern: int = -3
    weight_code_block: int = 5
    weight_very_long: int = 3       # >= 80 words
    weight_long: int = 2            # >= 35 words
    weight_requirement_list: int = 2  # per bullet / numbered / list marker (max 3)

    # Score thresholds are now per-route (min_score / max_score), so
    # these two are kept only for backward-compatible label mapping when
    # a caller still uses the old "complexity label" API.
    complex_threshold: int = 7
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
    # ── Development actions ──
    r"\bimplement(a|ar|ación|ation)?\b",
    r"\bdesarroll(a|ar|o)\b",
    r"\bprogram(a|ar|ación)\b",
    r"\bcodea?(r|ndo)?\b",
    r"\bescrib(e|ir|iendo)\b.{0,20}(código|code|script)\b",
    r"\brefactor\b",
    r"\bdebug\b",
    r"\bbug\b",
    r"\btest(s|ing|ear)?\b",
    r"\bunit test\b",
    r"\bdeploy\b",
    r"\bdesplieg(a|ue|ar)\b",
    r"\bpublic(a(r|rá|ré|lo|la|ción|do|da)?)\b",
    r"\bproduction\b",
    r"\bproducción\b",
    r"\barchitecture\b",
    r"\barquitectura\b",
    r"\bplugin\b",
    r"\bgithub\b",
    r"\bpull request\b|\bPR\b",
    r"\bcodebase\b",
    r"\brepo(sitorio|sitory)?\b",
    r"\bbase de (código|datos)\b",
    r"\bend[- ]?to[- ]?end\b",
    r"\bcomplet(a|o)\b.{0,15}(app|aplicación|sistema|proyecto)\b",
    r"\bautomatiz(a|ar|ación)\b",
    r"\bcompilador\b",
    r"\bcompiler\b",
    r"\bdesde cero\b",
    r"\bfrom scratch\b",
    r"\bbuild\b",
    # ── Creation verbs with tech context ──
    r"\bcre(a|ar|ame|alo|ala)\b.{0,15}(app|aplicación|sistema|proyecto|web|software|servicio|api)\b",
    r"\bhac(er|eme|eme|ele)\b.{0,15}(un|una)\b.{0,10}(app|aplicación|sistema|proyecto|web)\b",
    # ── App / platform references ──
    r"\bapp\b",
    r"\baplicaci[óo]n\b",
    r"\bandroid\b",
    r"\bios\b",
    r"\bm[óo]vil\b",
    r"\bmobile\b",
    r"\bweb app\b",
    r"\bbackend\b",
    r"\bback[- ]?end\b",
    r"\bfrontend\b",
    r"\bfront[- ]?end\b",
    r"\bfull[- ]?stack\b",
    r"\bfront\b",  # "front y backend" = complex
    r"\bserver\b",
    r"\bservidor\b",
    r"\bintegr(a|ar|ción|ation)\b",
    r"\bautenticaci[óo]n\b",
    r"\blogin\b.{0,10}(sistema|system|con)\b",
    # ── Databases & storage (knowledge terms → medium) ──
    # ── APIs & integration (knowledge terms → medium) ──
    # ── Infrastructure → keep only action/build infra ──
    # ── Finance & business (domain-specific) ──
    r"\bfinancier(o|a|os|as)\b",
    r"\bfinancial\b",
    r"\bbanc(o|a|os|as|ario|aria)\b",
    r"\bbanking\b",
    r"\btransacci[óo]n\b",
    r"\bpago\b",
    r"\bpayment\b",
    r"\bstripe\b",
    # ── Infrastructure (action/build only) ──
    r"\bdocker\b",
    r"\bkubernetes\b|\bk8s\b",
    r"\bmicroservic(io|e)\b",
    r"\bAWS\b|\bAzure\b|\bGCP\b",
    r"\bhosting\b",
    # ── Machine learning / AI ──
    r"\bmachine learning\b",
    r"\binteligencia artificial\b",
    r"\bmodelo\b.{0,15}(entrenar|train|fine[- ]?tun)\b",
    # ── DevOps / CI/CD ──
    r"\bpipeline\b",
    r"\bCI/CD\b",
    r"\bworkflow\b",
    r"\bmonitor(ing|eo|ear)?\b",
    # ── Real-time / advanced ──
    r"\breal[- ]?time\b",
    r"\bwebsocket\b",
    r"\bscalable\b|\bescalable\b",
    r"\bresponsive\b",
]

_DEFAULT_MEDIUM_PATTERNS = [
    r"\bexplica(r|me)?\b",
    r"\bexpl[ií]came\b",
    r"\bresume(n|ir)?\b",
    r"\bcompara(r)?\b",
    r"\bplan\b",
    r"\bdiseña(r)?\b",
    r"\banaliza(r)?\b",
    r"\bdescribe\b",
    r"\bdraft\b",
    r"\bmejora(r)?\b",
    r"\boptimiza(r)?\b",
    r"\bc[óo]mo\b.{0,10}(funciona|hacer|usar|configurar)\b",
    r"\bhow\b.{0,10}(does|to|do|can)\b",
    r"\bqu[ée]\b es\b",
    r"\bwhat is\b",
    r"\bdiferencia\b",
    r"\bdifference\b",
    r"\btutorial\b",
    r"\bgu[ií]a\b",
    r"\bejemplo\b",
    r"\bexample\b",
    r"\brecomiend(a|as|an)\b",
    r"\brecommend\b",
    r"\bmejores pr[áa]cticas\b",
    r"\bbest practices\b",
    r"\brevis(a|ar)?\b",
    r"\breview\b",
    r"\bcorregir\b",
    r"\bfix\b",
    r"\bbuscar\b",
    r"\bsearch\b",
    r"\bconvertir\b",
    r"\bconvert\b",
    r"\bformato\b",
    r"\bvalidar\b",
    r"\bvalidate\b",
    r"\bdocument(a|ar|ación)\b",
    r"\bopin(ión|as)\b",
    r"\bopini[oó]n\b",
    r"\bsuger(ir|encia)\b",
    r"\bsuggest\b",
    # ── Technology domain terms (knowledge / explanation) ──
    r"\bAPI\b",
    r"\bREST\b",
    r"\bGraphQL\b",
    r"\bendpoint\b",
    r"\bwebhook\b",
    r"\bOAuth\b",
    r"\bdatabase\b",
    r"\bbase de datos\b",
    r"\bSQL\b",
    r"\bMySQL\b",
    r"\bPostgreSQL\b",
    r"\bMongoDB\b",
    r"\bRedis\b",
    r"\bORM\b",
    r"\bschema\b",
    r"\bCRUD\b",
    r"\bJWT\b",
    r"\bencrypt\b",
    r"\bhash\b",
    r"\bcifrado\b",
    r"\bseguridad\b",
    r"\bsecurity\b",
    r"\bframework\b",
    r"\blibrer[ií]a\b",
    r"\blibrary\b",
    r"\bCLI\b",
    r"\bPDF\b",
    r"\bExcel\b",
    r"\bCSV\b",
    r"\bexport(ar)?\b",
    r"\bimport(ar)?\b",
    r"\bcach[eé]\b",
    r"\bscrap(e|ing|er)\b",
    r"\bchatbot\b",
    r"\bdataset\b",
    r"\blog\b.{0,10}(sistema|system|centralizado)\b",
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
    Route(name="simple", provider="nous", model="openrouter/owl-alpha",
          min_score=0, max_score=1, emoji="🟢"),
    Route(name="medium", provider="opencode-go", model="deepseek-v4-pro",
          min_score=2, max_score=6, emoji="🟡"),
    Route(name="complex", provider="openai-codex", model="gpt-5.5",
          min_score=7, max_score=999, emoji="🔴"),
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

    def _get_default_route(name: str) -> Route:
        """Look up the built-in default route by *name*, not by list index."""
        for r in DEFAULT_ROUTES:
            if r.name == name:
                return r
        # Last-resort fallback — should never happen with built-in names
        return DEFAULT_ROUTES[0]

    routes: List[Route] = []
    for name in ("simple", "medium", "complex"):
        entry = _to_mapping(raw.get(name))
        if not entry:
            legacy = _LEGACY_DEFAULTS[name]
            fallback = _get_default_route(name)
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
        fallback = _get_default_route(name)
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
        llm_classifier_enabled=bool(section.get("llm_classifier_enabled", True)),
        llm_classifier_provider=str(section.get("llm_classifier_provider", "nous")),
        llm_classifier_model=str(section.get("llm_classifier_model", "openrouter/owl-alpha")),
        routes=_load_routes(raw_routes),
        scoring=_load_scoring(raw_scoring),
        patterns=_load_patterns(raw_patterns),
    )
