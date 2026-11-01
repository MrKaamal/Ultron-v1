# Security policy

ULTRON is an alpha prototype. Its guardrails are described, with their known limits, in [docs/SAFETY.md](docs/SAFETY.md).

## Reporting a vulnerability

Please do **not** open a public issue for a sandbox escape, policy bypass, audit-log bypass or secret leak.
Use GitHub's private vulnerability reporting ("Security" tab → "Report a vulnerability") on this repository.
Include the ULTRON version, a minimal reproduction and what you expected to be blocked.

## Out of scope

Anything that requires you to enable `approval = "auto"` outside a container, run a skill you wrote, or approve a
prompt yourself is working as documented.
