# Providers

| `provider` | Needs | Default model | Notes |
|---|---|---|---|
| `auto` | | | `anthropic` if `ANTHROPIC_API_KEY` is set, `openai` if `OPENAI_API_KEY` **and** a model are set, else `mock` |
| `mock` | nothing | `offline-demo-brain` | scripted demo planner, not a language model |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | Messages API over HTTPS, no SDK |
| `openai` | `OPENAI_API_KEY`, `model` | none (you choose) | Chat Completions |
| `openai-compat` | `base_url`, `model`, optional `ULTRON_API_KEY` | none | LM Studio, vLLM, OpenRouter, Groq, ... |
| `ollama` | a running Ollama | `llama3.2:3b` | `http://localhost:11434/v1`, no key, never uses a proxy |

## Claude

```bash
export ANTHROPIC_API_KEY=...
ultron run "..."                                   # claude-sonnet-5-5
ultron run --model claude-opus-5-5 "..."           # any model ID your account can use
```

Optional: `ULTRON_ANTHROPIC_VERSION` (default `2026-10-21`), `ANTHROPIC_BASE_URL` (gateways/proxies),
`ANTHROPIC_WORKSPACE_ID` (keys not scoped to one workspace), `effort` in `ultron.toml`.

ULTRON never sends `temperature`, `top_p`, `top_k`, `tool_choice` or a `thinking` setting: current models reject some of
them and default to adaptive thinking. Thinking blocks the API returns are replayed untouched in tool loops.

## Ollama (fully local)

```bash
ollama serve &
ollama pull llama3.2:3b
ultron run --provider ollama "List the files in the workspace"
```

Small local models are much weaker at tool use than hosted ones: expect more malformed calls, which ULTRON reports
back to the model as errors. Try a larger model for real work; on an 8 GB laptop stay with 3B-8B quantized models.

## Any OpenAI-compatible server

```bash
ULTRON_BASE_URL=http://localhost:1234/v1 ULTRON_MODEL=my-model ultron run --provider openai-compat "..."
```

## Troubleshooting

* `HTTP 401 ... check your API key`: wrong or expired key (the key is never echoed in errors).
* `HTTP 404 ... check the model name and base URL`: the model ID is not available to your account.
* `cannot reach localhost:11434`: start Ollama (`ollama serve`).
* `HTTP 429` / `529`: retried automatically up to 3 times; check rate limits or try again later.
* Mismatch with a real API response? Open an issue with the (redacted) error line, which includes the request ID.
