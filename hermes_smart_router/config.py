"""Configuration loading for Hermes Smart Router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class Route:
    """A concrete provider/model route to apply to a Hermes gateway session."""

    name: str
    provider: str
    model: str
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


@dataclass(frozen=True)
class RouterConfig:
    enabled: bool
    dry_run: bool
    respect_manual_override: bool
    routes: Dict[str, Route]

    def route_for(self, complexity: str) -> Route:
        return self.routes.get(complexity) or self.routes["complex"]


DEFAULT_ROUTES: Dict[str, Route] = {
    "simple": Route(name="simple", provider="nous", model="deepseek-v4-free"),
    "medium": Route(name="medium", provider="opencode-go", model="deepseek-v4-pro"),
    "complex": Route(name="complex", provider="openai-codex", model="gpt-5.5"),
}


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    data = {}
    for key in ("enabled", "dry_run", "respect_manual_override", "routes"):
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


def _route_from_mapping(name: str, raw: Any, fallback: Route) -> Route:
    data = _to_mapping(raw)
    if not data:
        return fallback
    return Route(
        name=name,
        provider=str(data.get("provider") or fallback.provider),
        model=str(data.get("model") or fallback.model),
        api_key=data.get("api_key"),
        base_url=data.get("base_url"),
        api_mode=data.get("api_mode"),
    )


def load_config(gateway: Any = None) -> RouterConfig:
    """Load plugin config from `gateway.config.smart_router` if present.

    Defaults match Alfonso's target routing:
    - simple -> nous/deepseek-v4-free
    - medium -> opencode-go/deepseek-v4-pro
    - complex -> openai-codex/gpt-5.5
    """
    section = _get_smart_router_section(gateway)
    raw_routes = _to_mapping(section.get("routes"))

    routes = {
        name: _route_from_mapping(name, raw_routes.get(name), fallback)
        for name, fallback in DEFAULT_ROUTES.items()
    }

    return RouterConfig(
        enabled=bool(section.get("enabled", True)),
        dry_run=bool(section.get("dry_run", False)),
        respect_manual_override=bool(section.get("respect_manual_override", True)),
        routes=routes,
    )
