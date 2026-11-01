# Safety model

ULTRON assumes the language model is **useful but not trustworthy**: it can misunderstand, be tricked by text it
reads (prompt injection), or simply be wrong. So safety does not depend on the model behaving. It depends on code
the model cannot reach.

## Principles

1. **The model only requests; code decides.** Every tool call passes `Policy.evaluate` before it runs.
2. **Deny is final.** Approval settings (`ask`, `deny`, `auto`) only change what happens to calls that would have
   *asked*. No setting turns a denial into an allow.
3. **Fail closed.** No terminal to ask on → declined. Unattended (sentinel) → declined. Unknown tool → error.
   Audit log unreadable → the run stops.
4. **The cage is outside the cage.** State, inbox, outbox and skills folders must live outside the workspace. Startup
   refuses a configuration where they do not, so the agent has no tool that can reach its own guardrails.
5. **Least surprise.** Reading is free, writing asks, running code asks, network is off until you allowlist domains.

## Layers (a call must pass all of them)

| Layer | Checks | Result when it fails |
|---|---|---|
| Argument validation | JSON schema types, required fields | error result to the model |
| Policy check hook | per-tool: path resolution, command allow-list, URL rules | **deny** |
| Base risk | `safe` → allow, `confirm` → ask, `forbidden` → deny | ask / deny |
| Approval | human `y/n/always` (never "always" for `run_command` or `http_get`) | deny if declined |
| Kill switch | checked again right before executing | run stops, transcript stays valid |
| Tool itself | re-validates paths and commands (second line of defence) | deny |
| Output | secrets redacted, size capped | n/a |

## Files

* All paths are resolved against the workspace with symlinks followed; `..`, absolute paths and links pointing out are denied.
* Protected names are denied even when they do not exist yet: `.env`, `.env.*` (not `.example`), `*.pem`, `*.key`,
  `id_rsa*`, `id_ed25519*`, `.netrc`, `.npmrc`, `.pypirc`, `.git-credentials`, and the folders `.ssh`, `.aws`, `.gnupg`, `.git`.
* Writes are atomic (temp file + rename) and size-limited.

## Commands

* No shell: the command is split with `shlex` and run directly, so `|`, `>`, `&&`, `;`, `$(...)` and backticks are rejected.
* Only commands in `allowed_commands` run. The read-only set (`ls cat head tail wc grep echo pwd diff`) is allowed
  without asking; anything else, including `python3`, **asks**.
* Arguments that look like paths go through the sandbox (also `--flag=path` forms and glob expansions).
* Known-destructive patterns (`sudo`, `rm -rf`, `mkfs`, `dd if=`, fork bombs, `curl | sh`, ...) are denied outright.
* The child gets a minimal environment (no API keys), its own process group, a CPU/file-size limit and a timeout; the
  whole group is killed on timeout.

## Network

`http_get` is **disabled** until you list domains. Then: http/https only, no embedded credentials, host must match the
allowlist, every resolved address must be public (blocks `127.0.0.1`, `10.x`, `169.254.169.254`, IPv4-mapped IPv6, ...),
redirects are re-validated hop by hop, text content types only, size capped. Returned text is labelled untrusted.

## Prompt injection

Text from files, commands and web pages reaches the model, and the model may obey it. ULTRON limits the blast radius
rather than pretending to detect it: the system prompt labels tool output as data; secrets are redacted from output;
and anything with side effects still goes through policy and approval. If an injected instruction asks the agent to
read `.env` or write outside the workspace, the policy denies it, and `max_denials` ends the run.

## Audit log

`state/audit.jsonl` is append-only JSON Lines. Each record contains the SHA-256 of the previous record plus its own
body, so editing, deleting or reordering a record in the middle is detected by `ultron audit --verify`. Secrets are
redacted and long values truncated before writing.

**What it does not give you:** if an attacker can rewrite the whole file they can rebuild a valid chain, and removing
records from the *end* leaves a valid shorter chain. Pin the head hash (printed by `--verify`) somewhere separate if you
need to detect that.

## Kill switch

`ultron kill` creates `state/KILL`. Every agent step and every tool call checks it, including between approval and
execution. A killed run closes every open tool call with a `cancelled` result so the conversation can continue after
`ultron revive`. The sentinel exits when it sees the file. Tasks that were running when the switch was thrown are
moved to `inbox/failed/` rather than silently retried.

## Skills

Skills are Python executed at startup with your privileges, so they are *operator-trusted code*: the folder must be
outside the workspace, symlinks are not loaded, a skill cannot shadow a built-in tool, and every loaded file's SHA-256
is audited. Tools without an explicit `risk="safe"` ask for approval.

## Recommended setups

| Situation | Setup |
|---|---|
| Trying it out | `ultron demo`, then `ultron run` with defaults (`approval = ask`) |
| Daily use on your machine | defaults; keep the workspace a dedicated folder, not your home directory |
| Unattended / sentinel | Docker (`docker-compose.yml`), `approval = "auto"` only inside the container, a separate API key with a spend limit |
| Anything touching untrusted input | container or VM, `approval = "deny"`, no web domains |

## Reporting a vulnerability

See [SECURITY.md](../SECURITY.md).
