"""Slash command handlers for Hermes Smart Router.

Visual tables, model discovery wizard, and route management — all
accessible via ``/smart-router ...`` from Telegram or CLI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .config import (
    Route,
    RouterConfig,
    ScoringConfig,
    PatternConfig,
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
    """Render a table header with Unicode box-drawing.

    Args:
        cols: List of (label, width, alignment) tuples.
        style: "double" for ╔═╗ or "round" for ╭─╮.
    """
    box = _BOX
    if style == "round":
        tl, h, t, tr = box["tl"], box["h"], box["t"], box["tr"]
        sl, sv, sr = box["v"], box["v"], box["v"]
    else:
        tl, h, t, tr = box["stl"], box["sh"], box["st"], box["str"]
        sl, sv, sr = box["sv"], box["sv"], box["sv"]

    # Top border
    top = tl + t.join(h * (w + 2) for _, w, _ in cols) + tr
    # Header labels
    labels = sl + sv.join(f" {_pad(label, w, a)} " for label, w, a in cols) + sr
    # Separator
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
        ("Model", 22, "left"),
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
            (route.model, 22, "left"),
        ])
        lines.append(row)
        if idx < len(cfg.routes) - 1:
            lines.append(_table_separator(cols))

    lines.append(_table_footer(cols))
    lines.append("```")
    lines.append(f"• Total routes: **{len(cfg.routes)}**")
    lines.append(f"• Classifier: {'🤖 LLM' if cfg.llm_classifier_enabled else '📋 regex'}")
    lines.append(f"• Dry-run: {'🔬 ON' if cfg.dry_run else '🚀 OFF'}  |  Footer: {'✅ ON' if cfg.show_route_footer else '❌ OFF'}")
    return "\n".join(lines)


def format_weights_table(cfg: RouterConfig) -> str:
    """Render a visual table of scoring weights."""
    s = cfg.scoring
    cols: List[Tuple[str, int, str]] = [
        ("Weight Parameter", 28, "left"),
        ("Value", 6, "right"),
        ("Effect", 44, "left"),
    ]

    weights = [
        ("weight_complex_pattern", s.weight_complex_pattern,
         "Points per complex keyword matched (implement, deploy...)"),
        ("weight_medium_pattern", s.weight_medium_pattern,
         "Points per medium keyword matched (explain, analyze...)"),
        ("weight_simple_pattern", s.weight_simple_pattern,
         "Points per simple keyword (negative = reduces score)"),
        ("weight_code_block", s.weight_code_block,
         "Added if message contains code blocks (```)"),
        ("weight_very_long", s.weight_very_long,
         "Added if message has ≥80 words"),
        ("weight_long", s.weight_long,
         "Added if message has ≥35 words"),
        ("weight_requirement_list", s.weight_requirement_list,
         "Per bullet/numbered/list marker (max 3)"),
    ]

    lines = [
        "**Smart Router — Scoring Weights**",
        "```",
        _table_header(cols, style="double"),
    ]

    for idx, (name, value, desc) in enumerate(weights):
        row = _table_row([
            (name, 28, "left"),
            (str(value), 6, "right"),
            (desc, 44, "left"),
        ])
        lines.append(row)
        if idx < len(weights) - 1:
            lines.append(_table_separator(cols))

    lines.append(_table_footer(cols))
    lines.append("```")
    lines.append("")
    lines.append("**Thresholds:**")
    lines.append(f"• Complex: score ≥ **{s.complex_threshold}**  |  Medium: score ≥ **{s.medium_threshold}**")
    return "\n".join(lines)


def format_patterns_table(cfg: RouterConfig) -> str:
    """Render the regex patterns for each complexity tier."""
    p = cfg.patterns

    lines = ["**Smart Router — Regex Patterns**"]

    for tier, emoji, patterns_list in [
        ("complex", "🔴", p.complex),
        ("medium", "🟡", p.medium),
        ("simple", "🟢", p.simple),
    ]:
        lines.append(f"\n{emoji} **{tier.upper()}** ({len(patterns_list)} patterns)")
        for pat in patterns_list[:12]:  # Show first 12, then summarize
            lines.append(f"  • `{pat}`")
        if len(patterns_list) > 12:
            lines.append(f"  • ... and {len(patterns_list) - 12} more")

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
        # Hermes config objects often expose dict-like access
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


def generate_wizard_guide(gateway: Any) -> str:
    """Generate a step-by-step configuration wizard as a chat message.

    The wizard:
    1. Shows available Hermes models
    2. Provides ready-to-copy YAML snippets
    3. Guides the user through adding routes to config.yaml
    """
    cfg = load_config(gateway)
    available_providers = _infer_available_providers(gateway)
    available_models = discover_hermes_models(gateway)

    lines = [
        "🧙 **Smart Router Setup Wizard**",
        "",
        "This wizard will help you configure your Smart Router routes.",
        "You'll need to edit `~/.hermes/config.yaml` — but don't worry, I'll give you",
        "ready-to-paste YAML snippets.",
        "",
    ]

    # Step 0: Show available models
    lines.append("───  📋  Step 1: Available Models  ───")
    lines.append("")
    lines.append("Here are the models already configured in your Hermes:")

    if available_models:
        for m in available_models:
            src = {"primary": "⭐ primary", "fallback": "↩️  fallback", "custom": "⚙️  custom"}.get(m["source"], m["source"])
            lines.append(f"  • `{m['provider']}/{m['model']}` ({src})")
    else:
        lines.append("  _(no models detected — run `hermes model` first)_")

    lines.append("")
    lines.append("**Edit your config:** `hermes config edit`  or  `nano ~/.hermes/config.yaml`")
    lines.append("")

    # Step 1: Minimal config example
    lines.append("───  🎯  Step 2: Add routes to config.yaml  ───")
    lines.append("")
    lines.append("Add a `smart_router` section with a `routes` list:")

    # Generate example based on detected providers
    if len(available_models) >= 3:
        r0, r1, r2 = available_models[0], available_models[1], available_models[2]
    else:
        r0 = {"provider": "nous", "model": "deepseek-v4-free"}
        r1 = {"provider": "opencode-go", "model": "deepseek-v4-pro"}
        r2 = {"provider": "openai-codex", "model": "gpt-5.5"}

    example_yaml = [
        "```yaml",
        "smart_router:",
        "  enabled: true",
        "  show_route_footer: true",
        "  dry_run: false",
        "  llm_classifier_enabled: false",
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
    lines.append("───  ⚖️  Step 3: Customize scoring (optional)  ───")
    lines.append("")
    lines.append("You can tweak the scoring weights under `smart_router.scoring`:")

    scoring_yaml = [
        "```yaml",
        "smart_router:",
        "  scoring:",
        "    weight_complex_pattern: 4",
        "    weight_medium_pattern: 2",
        "    weight_simple_pattern: -3",
        "    weight_code_block: 5",
        "    weight_very_long: 3",
        "    weight_long: 2",
        "    weight_requirement_list: 1",
        "    complex_threshold: 6",
        "    medium_threshold: 2",
        "```",
    ]
    lines.extend(scoring_yaml)

    lines.append("")
    lines.append("───  🏷️  Step 4: Customize patterns (optional)  ───")
    lines.append("")
    lines.append("Add custom regex patterns for each tier under `smart_router.patterns`:")

    patterns_yaml = [
        "```yaml",
        "smart_router:",
        "  patterns:",
        "    complex:",
        '      - "\\bimplement(a|ar)?\\b"',
        '      - "\\bdeploy\\b"',
        '      - "\\bplugin\\b"',
        "    medium:",
        '      - "\\bexplica(r|me)?\\b"',
        '      - "\\banaliza(r)?\\b"',
        "    simple:",
        '      - "^(ok|dale|gracias|hola)"',
        "```",
    ]
    lines.extend(patterns_yaml)

    lines.append("")
    lines.append("───  ✅  Step 5: Apply & verify  ───")
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
        "• `/smart-router weights` — show scoring weights\n"
        "• `/smart-router patterns` — show regex patterns\n"
        "• `/smart-router metrics` — show route usage metrics\n"
        "• `/smart-router version` — show plugin version\n\n"
        "🧪 **Test & Classify**\n"
        "• `/smart-router test <message>` — full classification breakdown\n"
        "• `/smart-router classifier <message>` — classify a message\n"
        "• `/smart-router dry-run <message>` — preview which route would be used\n\n"
        "⚙️ **Configure**\n"
        "• `/smart-router wizard` — step-by-step setup guide\n"
        "• `/smart-router dry-run on|off` — toggle dry-run mode\n"
        "• `/smart-router footer on|off` — toggle route footer\n"
        "• `/smart-router classifier llm|regex` — switch classifier mode\n"
        "• `/smart-router routes add-snippet <name> <min> <max> <provider> <model> [emoji]` — generate config snippet\n\n"
        "📖 **Help**\n"
        "• `/smart-router help` — this message"
    )


# ---------------------------------------------------------------------------
# Route snippet generator (for "add" command)
# ---------------------------------------------------------------------------

def generate_add_route_snippet(
    name: str,
    min_score: str,
    max_score: str,
    provider: str,
    model: str,
    emoji: str = "",
    gateway: Any = None,
) -> str:
    """Generate a ready-to-paste YAML snippet for adding a new route.

    Also validates that *provider* is available in Hermes config (if gateway provided).
    """
    try:
        mins = int(min_score)
    except ValueError:
        mins = 0
    try:
        maxs = int(max_score)
    except ValueError:
        maxs = 999

    display_emoji = emoji if emoji else {"simple": "🟢", "medium": "🟡", "complex": "🔴"}.get(name.lower(), "⚪")

    lines = [
        f"**Add this route to `~/.hermes/config.yaml`**",
        "",
        "```yaml",
        "smart_router:",
        "  routes:",
        f'    - name: "{name}"',
        f"      emoji: \"{display_emoji}\"",
        f"      min_score: {mins}",
        f"      max_score: {maxs}",
        f'      provider: "{provider}"',
        f'      model: "{model}"',
        "```",
        "",
    ]

    # Validate provider availability
    if gateway is not None:
        available = discover_hermes_models(gateway)
        matching = [m for m in available if m["provider"] == provider]
        if matching:
            lines.append(f"✅ Provider `{provider}` is available in your Hermes config")
            lines.append(f"   Model(s): {', '.join(m['model'] for m in matching)}")
        else:
            lines.append(f"⚠️  Provider `{provider}` was NOT found in your Hermes config.")
            lines.append(f"   Make sure it's configured or the route won't work.")
            if available:
                lines.append(f"   Available providers: {', '.join(set(m['provider'] for m in available))}")

    lines.append("")
    lines.append("After adding, restart: `hermes gateway restart`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

def handle_slash_command(text: str, gateway: Any, source: Any, cfg: RouterConfig) -> Optional[str]:
    """Handle a /smart-router slash command.

    Returns a string response to send, or None if not a command.
    """
    if not text.startswith("/smart-router"):
        return None

    parts = text.split(maxsplit=1)
    subcommand = parts[1].strip() if len(parts) > 1 else ""

    # ── Status ──
    if subcommand == "status" or subcommand == "":
        return format_routes_table(cfg)

    # ── Routes ──
    if subcommand == "routes":
        return format_routes_table(cfg)

    # ── Routes add-snippet ──
    if subcommand.startswith("routes add-snippet"):
        args = subcommand[len("routes add-snippet"):].strip().split()
        if len(args) < 5:
            return (
                "Usage: `/smart-router routes add-snippet <name> <min_score> <max_score> <provider> <model> [emoji]`\n\n"
                "Example: `/smart-router routes add-snippet premium 8 999 openai-codex gpt-5.5 🔴`\n\n"
                "This generates a YAML snippet you can paste into `~/.hermes/config.yaml`."
            )
        name = args[0]
        mins = args[1]
        maxs = args[2]
        prov = args[3]
        mod = args[4]
        emoji = args[5] if len(args) > 5 else ""
        return generate_add_route_snippet(name, mins, maxs, prov, mod, emoji, gateway)

    # ── Routes remove-snippet ──
    if subcommand.startswith("routes remove-snippet"):
        route_name = subcommand[len("routes remove-snippet"):].strip()
        if not route_name:
            route_names = [r.name for r in cfg.routes]
            return (
                "Usage: `/smart-router routes remove-snippet <name>`\n\n"
                f"Current routes: {', '.join(route_names)}\n\n"
                "This shows instructions for removing a route from `~/.hermes/config.yaml`."
            )
        # Find the route
        target = None
        for r in cfg.routes:
            if r.name.lower() == route_name.lower():
                target = r
                break
        if target is None:
            route_names = [r.name for r in cfg.routes]
            return f"❌ Route `{route_name}` not found. Current routes: {', '.join(route_names)}"
        return (
            f"**Remove route `{target.name}`**\n\n"
            f"Edit `~/.hermes/config.yaml` and delete this route from the `smart_router.routes` list:\n\n"
            f"```yaml\n{_generate_route_yaml(target)}\n```\n\n"
            f"After removing, restart: `hermes gateway restart`"
        )

    # ── Models ──
    if subcommand == "models":
        return format_models_table(gateway)

    # ── Weights ──
    if subcommand == "weights":
        return format_weights_table(cfg)

    # ── Patterns ──
    if subcommand == "patterns":
        return format_patterns_table(cfg)

    # ── Wizard ──
    if subcommand == "wizard":
        return generate_wizard_guide(gateway)

    # ── Help ──
    if subcommand == "help":
        return format_help()

    # ── Test (pass-through to router.py) ──
    if subcommand.startswith("test"):
        return None  # handled by router.py toggle section

    # ── Metrics (pass-through to router.py) ──
    if subcommand == "metrics":
        return None  # handled by router.py toggle section

    # ── Version (pass-through to router.py) ──
    if subcommand == "version":
        return None  # handled by router.py toggle section

    return None
