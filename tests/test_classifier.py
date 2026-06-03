"""Tests for LLM-based classifier (mocked API calls)."""

from unittest.mock import patch, Mock
from hermes_smart_router.classifier import classify_message, Classification


def _mock_llm_response(complexity: str, score: int, reason: str = "mock"):
    """Create a mock LLM API response."""
    import json
    mock_resp = Mock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": json.dumps({
                    "complexity": complexity,
                    "score": score,
                    "reason": reason,
                })
            }
        }]
    }).encode("utf-8")
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=False)
    return mock_resp


@patch("urllib.request.urlopen")
@patch("hermes_smart_router.classifier._resolve_api_key", return_value="fake-key")
def test_simple_greeting(mock_key, mock_urlopen):
    mock_urlopen.return_value = _mock_llm_response("simple", 0, "just a greeting")
    result = classify_message("Hola", provider="nous", model="test")
    assert result is not None
    assert result.complexity == "simple"
    assert result.score == 0


@patch("urllib.request.urlopen")
@patch("hermes_smart_router.classifier._resolve_api_key", return_value="fake-key")
def test_medium_explanation(mock_key, mock_urlopen):
    mock_urlopen.return_value = _mock_llm_response("medium", 4, "explanation request")
    result = classify_message("Explícame cómo funciona esto", provider="nous", model="test")
    assert result is not None
    assert result.complexity == "medium"


@patch("urllib.request.urlopen")
@patch("hermes_smart_router.classifier._resolve_api_key", return_value="fake-key")
def test_complex_implementation(mock_key, mock_urlopen):
    mock_urlopen.return_value = _mock_llm_response("complex", 8, "implementation task")
    result = classify_message("Implementa un plugin para Hermes con tests", provider="nous", model="test")
    assert result is not None
    assert result.complexity == "complex"


@patch("hermes_smart_router.classifier._resolve_api_key", return_value=None)
def test_no_api_key_returns_none(mock_key):
    result = classify_message("Hola", provider="nous", model="test")
    assert result is None


def test_empty_message_returns_none():
    result = classify_message("")
    assert result is None


def test_whitespace_only_returns_none():
    result = classify_message("   ")
    assert result is None
