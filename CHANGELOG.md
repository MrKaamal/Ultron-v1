# Changelog

## 0.1.0 (initial prototype)

- Agent loop with step, tool-call, time and denial budgets, repeat detection and kill switch.
- Policy engine (allow / ask / deny), workspace sandbox, secret redaction.
- Tools: `list_dir`, `read_file`, `write_file`, `search_files`, `run_command`, `http_get`, `calc`, `remember`, `recall`, `finish`.
- Providers: Claude (Messages API), OpenAI-compatible (OpenAI, Ollama, LM Studio, vLLM, ...), offline demo brain.
- SQLite memory, hash-chained audit log, sentinel (inbox/outbox) mode, skills plugin loader.
- CLI: `run chat tools memory history audit status kill revive sentinel doctor init demo`.
- Docker setup, CI workflow, 400+ tests.
