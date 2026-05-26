"""Directory-plugin entrypoint for Hermes Smart Router.

Hermes directory plugins are loaded from the repository root when installed via
`hermes plugins install owner/repo` or copied into `~/.hermes/plugins/`.
The actual implementation lives in the importable `hermes_smart_router` package
so the same code also works as a pip entry-point plugin.
"""

try:
    # Directory plugin load path: hermes_plugins.<plugin_slug>
    from .hermes_smart_router import register
except ImportError:  # pragma: no cover - pytest/import-from-repo-root path
    # Plain Python import path, e.g. pytest collecting this root __init__.py
    from hermes_smart_router import register

__all__ = ["register"]
