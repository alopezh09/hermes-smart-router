# hermes-smart-router

Dynamic and intelligent model router plugin for [Hermes Agent](https://hermes-agent.nousresearch.com/). It classifies incoming gateway messages and switches Hermes models on the fly.

## What it does

`hermes-smart-router` registers a `pre_gateway_dispatch` hook. Before a gateway message reaches the agent, the plugin classifies the message as:

- `simple` -> `nous` / `deepseek-v4-free`
- `medium` -> `opencode-go` / `deepseek-v4-pro`
- `complex` -> `openai-codex` / `gpt-5.5`

The first MVP uses deterministic local heuristics, so routing itself does not spend model tokens. Later versions can add optional cheap-LLM classification.

## Support scope

- ✅ Gateway inputs: Telegram, Discord, Slack, webhooks, and any other platform that goes through Hermes gateway dispatch.
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

For editable Hermes plugin discovery, either install with `hermes plugins install ...`, copy/symlink the repo into `~/.hermes/plugins/hermes-smart-router`, or rely on the pip entry point if Hermes is running in the same Python environment where this package is installed.

## Configuration

Add this optional section to `~/.hermes/config.yaml`:

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
