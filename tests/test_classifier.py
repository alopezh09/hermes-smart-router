from hermes_smart_router.classifier import classify_message


def test_simple_ack():
    result = classify_message("dale gracias")
    assert result.complexity == "simple"


def test_medium_explanation():
    result = classify_message("Explícame la diferencia entre fallback y routing")
    assert result.complexity in {"medium", "complex"}


def test_complex_plugin_work():
    result = classify_message("Implementa un plugin para Hermes con tests, GitHub Actions y manejo de errores")
    assert result.complexity == "complex"
