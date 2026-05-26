# hermes-smart-router

Dynamic and intelligent model router plugin for [Hermes Agent](https://hermes-agent.nousresearch.com/). It classifies incoming gateway messages and switches Hermes models on the fly.

## What it does

`hermes-smart-router` registers a `pre_gateway_dispatch` hook. Before a gateway message reaches the agent, the plugin classifies the message by complexity and routes it to the appropriate model:

- `simple` (score 0-1) -> `nous` / `deepseek-v4-free`
- `medium` (score 2-5) -> `opencode-go` / `deepseek-v4-pro`
- `complex` (score 6+) -> `openai-codex` / `gpt-5.5`

All defaults are fully parametrizable — you can customize routes, score ranges, weights, and regex patterns from `config.yaml`.

## Support scope

- ✅ **Currently tested and supported: Telegram gateway only.**
- ⚠️ Other Hermes gateway platforms such as Discord, Slack, webhooks, etc. are not officially supported yet, even if parts of the hook design may be reusable later.
- ⚠️ Direct CLI conversations are not routed yet by this plugin.
- ✅ Fail-open: if the plugin cannot classify or cannot access Hermes gateway internals, Hermes continues with its normal configured model/fallbacks.

## Install from GitHub

```bash
hermes plugins install alopezh09/hermes-smart-router --enable
hermes gateway restart
```

Alternative editable development install:

```bash
git clone https://github.com/alopezh09/hermes-smart-router.git
cd hermes-smart-router
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Slash Commands (Visual Configuration)

You don't have to edit `config.yaml` manually. Use these in-chat commands:

| Command | Description |
|---------|-------------|
| `/smart-router` | Visual status + routes table |
| `/smart-router routes` | Detailed route table with score ranges |
| `/smart-router models` | Discover available Hermes models (from your config) |
| `/smart-router wizard` | Step-by-step interactive setup guide |
| `/smart-router weights` | View scoring weights table |
| `/smart-router patterns` | View regex patterns per tier |
| `/smart-router routes add-snippet <name> <min> <max> <provider> <model> [emoji]` | Generate a ready-to-paste YAML snippet |
| `/smart-router routes remove-snippet <name>` | Generate instructions to remove a route |
| `/smart-router dry-run on\|off` | Toggle dry-run mode |
| `/smart-router footer on\|off` | Toggle route footer |
| `/smart-router classifier llm\|regex` | Switch classifier mode |
| `/smart-router classifier <message>` | Classify a message |
| `/smart-router dry-run <message>` | Preview which route would be used |
| `/smart-router help` | Full command reference |

## Configuration

Add this optional section to `~/.hermes/config.yaml`:

### Minimal (legacy format, backward-compatible)

```yaml
smart_router:
  enabled: true
  dry_run: false
  respect_manual_override: true
  routes:
    simple:
      provider: nous
      model: deepseek-v4-free
    medium:
      provider: opencode-go
      model: deepseek-v4-pro
    complex:
      provider: openai-codex
      model: gpt-5.5
```

### Full (parametrizable format with custom score ranges)

```yaml
smart_router:
  enabled: true
  dry_run: false
  show_route_footer: true
  routes:
    - name: cheap
      emoji: "🟢"
      min_score: 0
      max_score: 1
      provider: nous
      model: deepseek-v4-free

    - name: standard
      emoji: "🟡"
      min_score: 2
      max_score: 5
      provider: opencode-go
      model: deepseek-v4-pro

    - name: premium
      emoji: "🔴"
      min_score: 6
      max_score: 999
      provider: openai-codex
      model: gpt-5.5

  # Optional: custom scoring weights
  scoring:
    weight_complex_pattern: 4
    weight_medium_pattern: 2
    weight_simple_pattern: -3
    weight_code_block: 5
    weight_very_long: 3
    weight_long: 2
    weight_requirement_list: 1

  # Optional: custom regex patterns
  patterns:
    complex:
      - "implement(a|ar)?"
      - "deploy"
      - "plugin"
    medium:
      - "explica(r|me)?"
      - "analiza(r)?"
    simple:
      - "^(ok|dale|gracias|hola)"
```

See `examples/config.yaml` for the complete reference.

Do **not** put provider secrets in this repo. Keep API keys/OAuth credentials in Hermes auth, `.env`, or provider config.

## Safety behavior

This plugin reuses Hermes gateway's existing session model override mechanism. It does not edit Hermes core files and does not rewrite `config.yaml` per message.

Because that mechanism is currently internal to Hermes, the plugin uses defensive checks and fails open on unsupported Hermes versions.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## License

MIT
