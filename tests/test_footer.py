"""Tests for the transform_llm_output footer hook and decisions store."""

import time
from types import SimpleNamespace

from hermes_smart_router.router import (
    transform_llm_output,
    _get_decisions_store,
    _DECISION_MAX_AGE_S,
)


class FakeGateway:
    def __init__(self):
        self.config = {}
        self._smart_router_decisions = {}


def test_footer_appends_when_decision_exists():
    gateway = FakeGateway()
    gateway._smart_router_decisions["sess-1"] = {
        "ts": time.time(),
        "info": {
            "complexity": "simple",
            "provider": "nous",
            "model": "deepseek-v4-free",
            "score": 0,
            "emoji": "🟢",
            "reason": "test",
        },
    }

    result = transform_llm_output(
        response_text="Hola!",
        session_id="sess-1",
        gateway=gateway,
    )

    assert result is not None
    assert "Hola!" in result
    assert "Smart Router: simple" in result
    assert "nous/deepseek-v4-free" in result
    assert "🟢" in result


def test_footer_returns_none_without_gateway():
    result = transform_llm_output(
        response_text="Hola!",
        session_id="sess-1",
    )
    assert result is None


def test_footer_returns_none_with_no_decision():
    gateway = FakeGateway()
    result = transform_llm_output(
        response_text="Hola!",
        session_id="no-such-session",
        gateway=gateway,
    )
    assert result is None


def test_footer_cleans_up_decision_after_reading():
    gateway = FakeGateway()
    gateway._smart_router_decisions["sess-1"] = {
        "ts": time.time(),
        "info": {
            "complexity": "medium",
            "provider": "opencode-go",
            "model": "deepseek-v4-pro",
            "score": 5,
            "emoji": "🟡",
            "reason": "test",
        },
    }

    transform_llm_output(
        response_text="Explicación...",
        session_id="sess-1",
        gateway=gateway,
    )

    # Decision should be popped (one-shot)
    assert "sess-1" not in gateway._smart_router_decisions


def test_footer_respects_show_route_footer_false():
    """Footer should return None when show_route_footer is disabled."""
    gateway = FakeGateway()
    gateway._smart_router_runtime_config = {"show_route_footer": False}
    gateway._smart_router_decisions["sess-1"] = {
        "ts": time.time(),
        "info": {"complexity": "simple", "provider": "n",
                 "model": "m", "score": 0, "emoji": "🟢", "reason": "x"},
    }

    result = transform_llm_output(
        response_text="Hola!",
        session_id="sess-1",
        gateway=gateway,
    )
    assert result is None
    # Decision should still be in store (not popped when footer disabled)
    assert "sess-1" in gateway._smart_router_decisions


def test_empty_response_text_returns_none():
    gateway = FakeGateway()
    result = transform_llm_output(
        response_text="",
        session_id="sess-1",
        gateway=gateway,
    )
    assert result is None


def test_stale_decisions_are_purged():
    """Decisions older than _DECISION_MAX_AGE_S are removed."""
    gateway = FakeGateway()

    # Fresh decision
    gateway._smart_router_decisions["fresh"] = {
        "ts": time.time(),
        "info": {"complexity": "simple", "provider": "x",
                 "model": "y", "score": 0, "emoji": "🟢", "reason": "r"},
    }

    # Stale decision
    gateway._smart_router_decisions["stale"] = {
        "ts": time.time() - _DECISION_MAX_AGE_S - 60,
        "info": {"complexity": "simple", "provider": "x",
                 "model": "y", "score": 0, "emoji": "🟢", "reason": "r"},
    }

    # Accessing the store should trigger cleanup
    store = _get_decisions_store(gateway)
    assert "fresh" in store
    assert "stale" not in store


def test_store_has_timestamp():
    """Decisions stored via the helper include a timestamp."""
    from hermes_smart_router.router import _store_decision

    gateway = FakeGateway()
    _store_decision(gateway, "sess-x", {
        "complexity": "complex",
        "provider": "openai-codex",
        "model": "gpt-5.5",
        "score": 8,
        "emoji": "🔴",
        "reason": "build task",
    })

    entry = gateway._smart_router_decisions.get("sess-x")
    assert entry is not None
    assert "ts" in entry
    assert "info" in entry
    assert entry["info"]["complexity"] == "complex"
