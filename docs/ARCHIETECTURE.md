# Architecture

```text
src/ultron/
  cli.py          argparse front end (run, chat, tools, memory, audit, kill, sentinel, doctor, demo, init)
  runtime.py      build_runtime(cfg, approver, ...) -> Runtime: wires every component
  agent.py        the loop; RunResult; context compaction
  policy.py       allow / ask / deny decisions
  sandbox.py      workspace confinement and protected names
  config.py       defaults < ultron.toml < env < flags, plus validation
  messages.py     provider-neutral Message / ToolCall / ToolResult / LLMResponse
  prompts.py      system prompt (static) and per-run user message (context + goal)
  providers/      base Protocol, anthropic.py, openai_compat.py (+Ollama), mock.py (offline brain)
  tools/          base.py (Tool, Risk, Verdict, coerce_args), fs, shell, web, calc, notes (+finish)
  memory.py       SQLite: notes and run history
  audit.py        hash-chained JSONL
  killswitch.py   file-based stop
  sentinel.py     inbox -> outbox runner
  skills.py       plugin loader
  netutil.py      urllib JSON POST with retries, proxy handling (loopback bypasses proxies)
  redact.py       secret scrubbing
  ui.py, demo.py  terminal output, approval prompt, guided tour
```

## One run

1. `Agent.run(goal)` searches memory for relevant notes and builds the first user message:
   current time, recalled notes (marked untrusted) and `OPERATOR GOAL:` + the goal.
2. Loop, at most `max_steps` times: check kill switch and time budget → `provider.complete(system, messages, tools)`.
3. No tool calls → the text is the answer. Otherwise each call goes through
   `validate → policy → (approval) → kill check → execute → redact → truncate`.
4. All results for a turn go back as **one** tool message (every tool call must be answered, even when the run is
   stopped half-way: unanswered calls get a `cancelled` result).
5. `finish(summary, status)` ends the run. Budgets, repeat detection, the denial limit, the kill switch and provider
   errors end it with `budget`, `stuck`, `blocked`, `killed` or `error`.
6. The run is recorded in memory and in the audit log (`run_start`, `llm`, `decision`, `approval`, `tool_end`, `run_end`).

## Providers

A provider implements `complete(system, messages, tools) -> LLMResponse`, raises `ProviderError` (never leaking
secrets) and may set `append_only = True`.

* **Anthropic** (`POST /v1/messages`): `x-api-key` + `anthropic-version` headers; tools as `{name, description,
  input_schema}`; results as `tool_result` blocks placed first in the user message. The full response content is kept
  in `Message.raw` and replayed verbatim, because thinking blocks must come back unmodified and are bound to the
  exact conversation prefix. For that reason the provider is `append_only` (no context compaction), and the system
  prompt never changes between turns (time and recalled notes live in the user message). Sampling parameters and
  forced `tool_choice` are never sent; current models reject them.
* **OpenAI-compatible** (`POST {base}/chat/completions`): OpenAI, Ollama (`http://localhost:11434/v1`), LM Studio,
  vLLM, OpenRouter, ... Handles argument strings that are not valid JSON, missing call ids and `<think>` blocks.
* **Mock**: a generator-based planner that replays against the recorded history each turn; stateless and deterministic.

HTTP is plain `urllib` with retries (408/429/5xx/529, `Retry-After` honoured and capped at 30 s, exponential
back-off) and loopback addresses never go through a proxy.

## Extending

* **A tool:** `@tool(description=..., risk="safe"|"confirm", parameters={...}, check=hook)` in a file under `skills/`.
  The `check` hook may return `Decision.deny/ask/allow` or raise `SandboxViolation` (a deny).
* **A provider:** implement the Protocol in `providers/base.py` and add a branch in `providers/__init__.py`.
* **A different approver** (Slack, web UI): any `Callable[[ApprovalRequest], bool]` passed to `build_runtime`.
* **Live events:** pass `reporter=callable(kind, data)` to `build_runtime`.

## Testing approach

No test needs the network or an API key. Providers run against a local `FakeServer` that records requests, so tests
assert the exact wire format. The agent runs with a `ScriptedProvider` that returns pre-written turns. The offline
demo is itself a smoke test (`ultron demo` exits non-zero if a guardrail misbehaves). Safety properties were
mutation-checked: removing the kill-switch check, the path-escape check, the audit hash check or thinking-block replay makes tests fail.
