"""Slash command handlers for Hermes Smart Router.

Visual tables, model discovery wizard, and route management — all
accessible via ``/smart-router ...`` from Telegram or CLI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .config import (
    Route,
    RouterConfig,
    load_config,
    DEFAULT_ROUTES,
)


# ---------------------------------------------------------------------------
# Unicode box-drawing helpers
# ---------------------------------------------------------------------------

_BOX = {
    "h": "─", "v": "│",
    "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
    "t": "┬", "b": "┴", "l": "├", "r": "┤", "x": "┼",
    "sh": "═", "sv": "║",
    "stl": "╔", "str": "╗", "sbl": "╚", "sbr": "╝",
    "st": "╦", "sb": "╩", "sl": "╠", "sr": "╣", "sx": "╬",
    "dash": "╌",
}


def _pad(text: str, width: int, align: str = "left") -> str:
    """Pad *text* to *width* with spaces."""
    if align == "center":
        return text.center(width)
    elif align == "right":
        return text.rjust(width)
    return text.ljust(width)


def _table_header(cols: List[Tuple[str, int, str]], style: str = "double") -> str:
    """Render a table header with Unicode box-drawing."""
    box = _BOX
    if style == "round":
        tl, h, t, tr = box["tl"], box["h"], box["t"], box["tr"]
        sl, sv, sr = box["v"], box["v"], box["v"]
    else:
        tl, h, t, tr = box["stl"], box["sh"], box["st"], box["str"]
        sl, sv, sr = box["sv"], box["sv"], box["sv"]

    top = tl + t.join(h * (w + 2) for _, w, _ in cols) + tr
    labels = sl + sv.join(f" {_pad(label, w, a)} " for label, w, a in cols) + sr
    sep = box["sl"] + box["sx"].join(box["sh"] * (w + 2) for _, w, _ in cols) + box["sr"]

    return f"{top}\n{labels}\n{sep}"


def _table_row(cols: List[Tuple[str, int, str]], style: str = "double") -> str:
    """Render a single table data row."""
    sv = _BOX["sv"] if style == "double" else _BOX["v"]
    return sv + sv.join(f" {_pad(text, w, a)} " for text, w, a in cols) + sv


def _table_footer(cols: List[Tuple[str, int, str]], style: str = "double") -> str:
    """Render table bottom border."""
    box = _BOX
    if style == "round":
        return box["bl"] + box["b"].join(box["h"] * (w + 2) for _, w, _ in cols) + box["br"]
    return box["sbl"] + box["sb"].join(box["sh"] * (w + 2) for _, w, _ in cols) + box["sbr"]


def _table_separator(cols: List[Tuple[str, int, str]], style: str = "double") -> str:
    """Render a mid-table separator row."""
    box = _BOX
    h = box["sh"] if style == "double" else box["h"]
    return box["sl"] + box["sx"].join(h * (w + 2) for _, w, _ in cols) + box["sr"]


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------

def format_routes_table(cfg: RouterConfig) -> str:
    """Render a visual Unicode table of all configured routes."""
    cols: List[Tuple[str, int, str]] = [
        ("Emoji", 5, "center"),
        ("Route", 12, "left"),
        ("Score", 10, "center"),
        ("Provider", 14, "left"),
        ("Model", 26, "left"),
    ]

    lines = [
        "**Smart Router — Routes**",
        "```",
        _table_header(cols, style="double"),
    ]

    for idx, route in enumerate(cfg.routes):
        score_range = f"{route.min_score}–{route.max_score}"
        row = _table_row([
            (route.emoji, 5, "center"),
            (route.name, 12, "left"),
            (score_range, 10, "center"),
            (route.provider, 14, "left"),
            (route.model, 26, "left"),
        ])
        lines.append(row)
        if idx < len(cfg.routes) - 1:
            lines.append(_table_separator(cols))

    lines.append(_table_footer(cols))
    lines.append("```")
    lines.append(f"• Total routes: **{len(cfg.routes)}**")
    lines.append(f"• Classifier: 🤖 **LLM** (`{cfg.llm_classifier_provider}/{cfg.llm_classifier_model}`)")
    lines.append(f"• Dry-run: {'🔬 ON' if cfg.dry_run else '🚀 OFF'}  |  Footer: {'✅ ON' if cfg.show_route_footer else '❌ OFF'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hermes model discovery
# ---------------------------------------------------------------------------

def discover_hermes_models(gateway: Any) -> List[Dict[str, str]]:
    """Extract available provider/model pairs from the Hermes gateway config.

    Returns a list of dicts with ``provider``, ``model``, and ``source`` keys.
    ``source`` indicates where the model was found:
    - ``primary`` — the main configured model
    - ``fallback`` — a fallback provider
    - ``custom`` — a custom provider
    """
    discovered: List[Dict[str, str]] = []
    seen: set = set()

    config = getattr(gateway, "config", None)
    if config is None:
        return discovered

    config_dict: Mapping[str, Any]
    if isinstance(config, Mapping):
        config_dict = config
    else:
        config_dict = {k: getattr(config, k, None) for k in dir(config) if not k.startswith("_")}

    def _add(provider: str, model: str, source: str) -> None:
        key = f"{provider}/{model}"
        if key not in seen:
            seen.add(key)
            discovered.append({"provider": provider, "model": model, "source": source})

    # 1. Primary model
    model_section = config_dict.get("model")
    if isinstance(model_section, Mapping):
        primary_provider = str(model_section.get("provider") or "")
        primary_model = str(model_section.get("default") or "")
        if primary_provider and primary_model:
            _add(primary_provider, primary_model, "primary")

    # 2. Fallback providers
    fallbacks = config_dict.get("fallback_providers", [])
    if isinstance(fallbacks, list):
        for fb in fallbacks:
            if isinstance(fb, Mapping):
                p = str(fb.get("provider") or "")
                m = str(fb.get("model") or "")
                if p and m:
                    _add(p, m, "fallback")

    # 3. Custom providers section
    providers_section = config_dict.get("providers")
    if isinstance(providers_section, Mapping):
        for prov_name, prov_cfg in providers_section.items():
            if isinstance(prov_cfg, Mapping):
                default_model = str(prov_cfg.get("default") or prov_cfg.get("model") or "")
                if default_model:
                    _add(str(prov_name), default_model, "custom")
            elif isinstance(prov_cfg, str):
                _add(str(prov_name), prov_cfg, "custom")

    return discovered


def format_models_table(gateway: Any) -> str:
    """Render a table of available Hermes models discovered from config."""
    models = discover_hermes_models(gateway)

    if not models:
        return (
            "**Available Hermes Models**\n\n"
            "No models detected from Hermes config.\n"
            "Check that your `~/.hermes/config.yaml` has `model.provider` and `model.default` set.\n\n"
            "Tip: run `hermes model` in your terminal to pick a model."
        )

    cols: List[Tuple[str, int, str]] = [
        ("Source", 10, "center"),
        ("Provider", 16, "left"),
        ("Model", 26, "left"),
    ]

    source_emoji = {"primary": "⭐", "fallback": "↩️ ", "custom": "⚙️ "}

    lines = [
        "**Available Hermes Models**",
        "_These are the models already configured in your Hermes. Use them in your Smart Router routes._",
        "```",
        _table_header(cols, style="double"),
    ]

    for idx, entry in enumerate(models):
        src = source_emoji.get(entry["source"], "  ")
        row = _table_row([
            (src, 10, "center"),
            (entry["provider"], 16, "left"),
            (entry["model"], 26, "left"),
        ])
        lines.append(row)
        if idx < len(models) - 1:
            lines.append(_table_separator(cols))

    lines.append(_table_footer(cols))
    lines.append("```")
    lines.append(f"• Total: **{len(models)}** model(s) found")
    lines.append("• Primary model is used as default; fallbacks are tried on failure")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wizard: step-by-step setup guide
# ---------------------------------------------------------------------------

def _infer_available_providers(gateway: Any) -> List[str]:
    """Return a flat list of provider names available in Hermes config."""
    models = discover_hermes_models(gateway)
    providers: List[str] = []
    seen: set = set()
    for m in models:
        if m["provider"] not in seen:
            seen.add(m["provider"])
            providers.append(m["provider"])
    return providers


def generate_wizard_guide(gateway: Any) -> str:
    """Generate a step-by-step configuration wizard as a chat message.

    The wizard:
    1. Shows available Hermes models
    2. Provides ready-to-copy YAML snippets
    3. Guides the user through adding routes to config.yaml
    """
    cfg = load_config(gateway)
    available_models = discover_hermes_models(gateway)

    lines = [
        "🧙 **Smart Router Setup Wizard**",
        "",
        "This wizard will help you configure your Smart Router routes.",
        "You'll need to edit `~/.hermes/config.yaml` — but don't worry, I'll give you",
        "ready-to-paste YAML snippets.",
        "",
        "**Classification:** All messages are classified via 🤖 **LLM** "
        f"(`{cfg.llm_classifier_provider}/{cfg.llm_classifier_model}`).",
        "Make sure your API key is set (see Step 2 below).",
        "",
    ]

    # Step 1: Show available models
    lines.append("───  📋  Step 1: Available Models  ───")
    lines.append("")

    if available_models:
        lines.append("Here are the models already configured in your Hermes:")
        for m in available_models:
            src = {"primary": "⭐ primary", "fallback": "↩️  fallback", "custom": "⚙️  custom"}.get(m["source"], m["source"])
            lines.append(f"  • `{m['provider']}/{m['model']}` ({src})")
    else:
        lines.append("  _(no models detected — run `hermes model` first)_")

    lines.append("")
    lines.append("**Edit your config:** `hermes config edit`  or  `nano ~/.hermes/config.yaml`")
    lines.append("")

    # Step 2: API key for LLM classifier
    lines.append("───  🔑  Step 2: API Key for LLM Classifier  ───")
    lines.append("")
    lines.append(f"The classifier uses `{cfg.llm_classifier_provider}` as its LLM provider.")
    lines.append("Make sure the corresponding API key is set:")
    provider_upper = cfg.llm_classifier_provider.upper().replace("-", "_")
    lines.append(f"```bash")
    lines.append(f"# Add to ~/.hermes/.env:")
    lines.append(f"{provider_upper}_API_KEY=your-key-here")
    lines.append(f"```")
    lines.append("")

    # Step 3: Routes config
    lines.append("───  🎯  Step 3: Add routes to config.yaml  ───")
    lines.append("")

    if len(available_models) >= 3:
        r0, r1, r2 = available_models[0], available_models[1], available_models[2]
    else:
        r0 = {"provider": "nous", "model": "openrouter/owl-alpha"}
        r1 = {"provider": "opencode-go", "model": "deepseek-v4-pro"}
        r2 = {"provider": "openai-codex", "model": "gpt-5.5"}

    example_yaml = [
        "```yaml",
        "smart_router:",
        "  enabled: true",
        "  show_route_footer: true",
        "  dry_run: false",
        f'  llm_classifier_provider: "{cfg.llm_classifier_provider}"',
        f'  llm_classifier_model: "{cfg.llm_classifier_model}"',
        "  routes:",
        f'    - name: "simple"',
        f"      emoji: \"🟢\"",
        f"      min_score: 0",
        f"      max_score: 1",
        f'      provider: "{r0["provider"]}"',
        f'      model: "{r0["model"]}"',
        "",
        f'    - name: "medium"',
        f"      emoji: \"🟡\"",
        f"      min_score: 2",
        f"      max_score: 5",
        f'      provider: "{r1["provider"]}"',
        f'      model: "{r1["model"]}"',
        "",
        f'    - name: "complex"',
        f"      emoji: \"🔴\"",
        f"      min_score: 6",
        f"      max_score: 999",
        f'      provider: "{r2["provider"]}"',
        f'      model: "{r2["model"]}"',
        "```",
    ]
    lines.extend(example_yaml)

    lines.append("")
    lines.append("───  ✅  Step 4: Apply & verify  ───")
    lines.append("")
    lines.append("After editing `config.yaml`, restart the gateway:")
    lines.append("```bash")
    lines.append("hermes gateway restart")
    lines.append("```")
    lines.append("Then verify with:")
    lines.append("• `/smart-router` — see your routes")
    lines.append("• `/smart-router dry-run hola` — test classification")
    lines.append("• `/smart-router models` — check available models")
    lines.append("")
    lines.append("🧙 **You're all set!** Your Smart Router is now configured.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

def format_help() -> str:
    """Return extended help text for all /smart-router commands."""
    return (
        "**Smart Router Commands**\n\n"
        "📊 **View & Inspect**\n"
        "• `/smart-router` — show status + route table\n"
        "• `/smart-router routes` — visual table of all routes\n"
        "• `/smart-router models` — discover available Hermes models\n"
        "• `/smart-router wizard` — step-by-step setup guide\n\n"
        "🔧 **Control**\n"
        "• `/smart-router dry-run on|off` — toggle dry-run mode\n"
        "• `/smart-router dry-run <message>` — preview classification\n"
        "• `/smart-router footer on|off` — toggle route footer\n\n"
        "🧪 **Test**\n"
        "• `/smart-router classifier <message>` — classify a message (LLM)\n\n"
        "📦 **Route Management**\n"
        "• `/smart-router routes add-snippet <name> <min> <max> <provider> <model> [emoji]`\n"
        "  Generate a ready-to-paste YAML snippet for a new route\n"
        "• `/smart-router routes remove-snippet <name>`\n"
        "  Generate instructions to remove a route\n\n"
        "⚙️ **Classification:** All messages are classified via 🤖 **LLM** "
        "(no regex fallback). Configure the LLM provider and model via\n"
        "`llm_classifier_provider` and `llm_classifier_model` in your\n"
        "`smart_router` config section."
    )


# ---------------------------------------------------------------------------
# Route snippet generators
# ---------------------------------------------------------------------------

def _generate_route_yaml(route: Route) -> str:
    """Generate a YAML snippet for a single route."""
    lines = [
        f'  - name: "{route.name}"',
        f"    emoji: \"{route.emoji}\"",
        f"    min_score: {route.min_score}",
        f"    max_score: {route.max_score}",
        f'    provider: "{route.provider}"',
        f'    model: "{route.model}"',
    ]
    if route.api_key:
        lines.append(f'    api_key: "{route.api_key}"')
    if route.base_url:
        lines.append(f'    base_url: "{route.base_url}"')
    if route.api_mode:
        lines.append(f'    api_mode: "{route.api_mode}"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main slash command dispatcher
# ---------------------------------------------------------------------------

def handle_slash_command(text: str, gateway: Any, source: Any, cfg: RouterConfig) -> Optional[str]:
    """Handle /smart-router slash commands. Returns response text or None."""

    if not text.startswith("/smart-router"):
        return None

    parts = text.split(maxsplit=1)
    subcommand = parts[1].strip() if len(parts) > 1 else ""

    # /smart-router or /smart-router status
    if subcommand == "" or subcommand == "status":
        return format_routes_table(cfg)

    # /smart-router routes
    if subcommand == "routes":
        return format_routes_table(cfg)

    # /smart-router models
    if subcommand == "models":
        return format_models_table(gateway)

    # /smart-router wizard
    if subcommand == "wizard":
        return generate_wizard_guide(gateway)

    # /smart-router help
    if subcommand == "help":
        return format_help()

    # /smart-router classifier <message>
    if subcommand.startswith("classifier"):
        arg = subcommand[len("classifier"):].strip()
        if arg and arg.lower() not in ("llm", "on", "true", "1", "enable", "regex", "off", "false", "0", "disable"):
            from .classifier import classify_message
            classification = classify_message(
                arg,
                provider=cfg.llm_classifier_provider,
                model=cfg.llm_classifier_model,
            )
            if classification is None:
                return (
                    "📋 **Smart Router classifier**\n\n"
                    f"⚠️ LLM classifier unavailable — could not classify message.\n"
                    f"Check that `{cfg.llm_classifier_provider.upper()}_API_KEY` is set."
                )
            route = cfg.route_for_score(classification.score)
            return (
                "📋 **Smart Router classifier**\n"
                f"{route.emoji} {route.name} → `{route.provider}/{route.model}` (score: {classification.score})\n"
                f"Mode: LLM ({cfg.llm_classifier_provider}/{cfg.llm_classifier_model})\n"
                f"Reason: {classification.reason}"
            )
        # Toggle commands handled by router.py
        return None

    # /smart-router routes add-snippet <name> <min> <max> <provider> <model> [emoji]
    if subcommand.startswith("routes add-snippet"):
        args_str = subcommand[len("routes add-snippet"):].strip()
        if not args_str:
            return (
                "Usage: `/smart-router routes add-snippet <name> <min> <max> <provider> <model> [emoji]`\n"
                "Example: `/smart-router routes add-snippet premium 6 999 openai-codex gpt-5.5 🔴`"
            )

        args = args_str.split()
        if len(args) < 5:
            return (
                "Usage: `/smart-router routes add-snippet <name> <min> <max> <provider> <model> [emoji]`\n"
                f"Got {len(args)} args, need at least 5."
            )

        try:
            name = args[0]
            min_s = int(args[1])
            max_s = int(args[2])
            provider = args[3]
            model = args[4]
            emoji = args[5] if len(args) > 5 else "⚪"
        except ValueError:
            return "Error: min_score and max_score must be integers."

        route = Route(name=name, provider=provider, model=model,
                      min_score=min_s, max_score=max_s, emoji=emoji)
        snippet = _generate_route_yaml(route)
        return (
            "**Add this to your `smart_router.routes` list in `~/.hermes/config.yaml`:**\n\n"
            "```yaml\n"
            f"{snippet}\n"
            "```\n\n"
            "Then restart the gateway:\n"
            "```bash\n"
            "hermes gateway restart\n"
            "```"
        )

    # /smart-router routes remove-snippet <name>
    if subcommand.startswith("routes remove-snippet"):
        arg = subcommand[len("routes remove-snippet"):].strip()
        if not arg:
            return "Usage: `/smart-router routes remove-snippet <name>`"

        return (
            f"**To remove the `{arg}` route:**\n\n"
            f"1. Edit `~/.hermes/config.yaml`\n"
            f"2. Find the route with `name: \"{arg}\"` under `smart_router.routes`\n"
            f"3. Delete that entire route block\n"
            f"4. Restart the gateway: `hermes gateway restart`"
        )

    return None
