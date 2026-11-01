# Contributing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && pytest
```

Ground rules:

1. **Safety changes need tests for the deny path**, not only the happy path.
2. New tools need a `risk` (default is `confirm`), a `check` hook that resolves paths through the sandbox, and an entry in the tools test.
3. Tests must not need the network or an API key. Use `ScriptedProvider` and `FakeServer` from `tests/helpers.py`.
4. Keep runtime dependencies at zero unless there is a very good reason.
5. Never loosen a guardrail without explaining why in `docs/SAFETY.md`.
